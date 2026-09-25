"""ModernBERT для вывода на CPU: те же веса и ответы, что у transformers, без матрицы внимания n × n в локальных слоях.

В ModernBERT два слоя из трёх смотрят только на соседей в окне ±sliding_window токенов,
но transformers считает их, как глобальные: полная матрица n × n и маска поверх неё.
На 8192 токенах это 8 лишних матриц по 6 × 8192 × 8192 чисел. Здесь локальный слой
режет текст на блоки по sliding_window токенов, и каждый блок смотрит только на себя
и двух соседей: работа растёт с длиной линейно.

Второе упрощение: классификатор берёт только токен CLS. Слоям после последнего
глобального нужны ответы лишь в начале текста (слою L − 1 — только позиция 0, слою
L − 2 — позиции 0…w и так далее), поэтому последний глобальный слой считает запросы
только для этого начала, а локальные слои после него работают на нём же.

Третье: глобальные слои считают запросы кусками. Оценки внимания n × n onnxruntime
держит в памяти целиком, и на 8192 токенах процесс занимал 3,8 ГБ; с 16 кусками в
памяти оценки одного куска, пик около 1 ГБ, а скорость та же. Это важно для слабых
ноутбуков и песочниц агентов, где памяти мало.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# На сколько кусков делить запросы глобального слоя: пик памяти на 8192 токенах 1 ГБ вместо 3,8.
GLOBAL_CHUNKS = 16


def bias(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Маска как слагаемое к оценкам: у запрещённых пар самое малое число, а не −inf.

    С −inf у запроса-паддинга без разрешённых ключей softmax дал бы NaN, и он утёк бы
    дальше через умножение на нулевые веса.
    """
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


class FastModernBert(nn.Module):
    """ModernBertForSequenceClassification с пулингом CLS: input_ids, attention_mask → logits."""

    def __init__(self, model: Any, chunks: int = GLOBAL_CHUNKS) -> None:
        super().__init__()
        self.chunks = chunks
        cfg = model.config
        if cfg.classifier_pooling != "cls":
            raise ValueError(f"FastModernBert умеет только пулинг cls, а не {cfg.classifier_pooling}")
        self.embeddings = model.model.embeddings
        self.layers = model.model.layers
        self.final_norm = model.model.final_norm
        self.head = model.head
        self.classifier = model.classifier
        self.types = list(cfg.layer_types)
        self.heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size // cfg.num_attention_heads
        self.window = int(cfg.sliding_window)
        for kind in set(self.types):
            theta = cfg.rope_parameters[kind]["rope_theta"]
            inv = 1.0 / (theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
            self.register_buffer(f"inv_{kind}", inv, persistent=False)
        self.last_global = max(i for i, t in enumerate(self.types) if t == "full_attention")
        # Сколько позиций начала нужно слоям после последнего глобального, чтобы токен CLS был точным.
        self.prefix = (len(self.types) - 1 - self.last_global) * self.window + 1

    def rotary(self, kind: str, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(n, device=device, dtype=torch.float32)
        freqs = pos[:, None] * getattr(self, f"inv_{kind}")[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def qkv(self, layer: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, _ = x.shape
        q, k, v = layer.attn.Wqkv(layer.attn_norm(x)).view(b, n, 3, self.heads, self.dim).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def out(self, layer: Any, h: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        b, _, n, _ = att.shape
        h = h + layer.attn.Wo(att.transpose(1, 2).reshape(b, n, self.heads * self.dim))
        return h + layer.mlp(layer.mlp_norm(h))

    def local(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Внимание в окне |i − j| ≤ w: блоки по w запросов смотрят на 3w ключей своего и соседних блоков."""
        b, h, n, d = q.shape
        w = self.window
        pad = (w - n % w) % w
        m = (n + pad) // w
        q = F.pad(q, (0, 0, 0, pad)).view(b, h, m, w, d)

        def around(x: torch.Tensor) -> torch.Tensor:
            x = F.pad(x, (0, 0, 0, pad)).view(b, h, m, w, d)
            x = F.pad(x, (0, 0, 0, 0, 1, 1))
            return torch.cat((x[:, :, :-2], x[:, :, 1:-1], x[:, :, 2:]), dim=3)

        k3, v3 = around(k), around(v)
        kb = F.pad(F.pad(keep, (0, pad)).view(b, m, w), (0, 0, 1, 1))
        keep3 = torch.cat((kb[:, :-2], kb[:, 1:-1], kb[:, 2:]), dim=2)
        # Запрос a блока смотрит на ключ c из трёх блоков: расстояние a − (c − w).
        a = torch.arange(w, device=q.device)[:, None]
        c = torch.arange(3 * w, device=q.device)[None, :]
        band = (a - c + w).abs() <= w
        allowed = band[None, None, None] & keep3[:, None, :, None, :]
        scores = (q @ k3.transpose(-1, -2)) * d**-0.5 + bias(allowed, q.dtype)
        att = torch.softmax(scores, dim=-1) @ v3
        return att.reshape(b, h, m * w, d)[:, :, :n]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        keep = attention_mask.bool()
        n = input_ids.shape[1]
        h = self.embeddings(input_ids=input_ids)
        cs = {kind: self.rotary(kind, n, h.device) for kind in set(self.types)}
        full = bias(keep[:, None, None, :], h.dtype)
        for i, layer in enumerate(self.layers):
            if i > self.last_global:
                break
            cos, sin = cs[self.types[i]]
            q, k, v = self.qkv(layer, h)
            q, k = rope(q, cos, sin), rope(k, cos, sin)
            if i == self.last_global:
                # Дальше нужны только позиции начала: запросы и остаток слоя — только для них.
                q, h = q[:, :, : self.prefix], h[:, : self.prefix]
            if self.types[i] == "full_attention" and i < self.last_global and self.chunks > 1:
                # Запросы по кускам: onnxruntime держит в памяти оценки одного куска, а не n × n.
                parts = q.tensor_split(self.chunks, dim=2)
                att = torch.cat([F.scaled_dot_product_attention(c, k, v, attn_mask=full) for c in parts], dim=2)
            elif self.types[i] == "full_attention":
                att = F.scaled_dot_product_attention(q, k, v, attn_mask=full)
            else:
                att = self.local(q, k, v, keep)
            h = self.out(layer, h, att)
        # Слои после последнего глобального — на начале текста, с маской окна.
        p = h.shape[1]
        near = (torch.arange(p, device=h.device)[:, None] - torch.arange(p, device=h.device)[None, :]).abs()
        tail = bias((near <= self.window)[None, None] & keep[:, None, None, : self.prefix], h.dtype)
        for i in range(self.last_global + 1, len(self.layers)):
            layer = self.layers[i]
            cos, sin = cs[self.types[i]]
            q, k, v = self.qkv(layer, h)
            q, k = rope(q, cos[:, :, :p], sin[:, :, :p]), rope(k, cos[:, :, :p], sin[:, :, :p])
            h = self.out(layer, h, F.scaled_dot_product_attention(q, k, v, attn_mask=tail))
        cls = self.final_norm(h[:, 0])
        return self.classifier(self.head(cls))
