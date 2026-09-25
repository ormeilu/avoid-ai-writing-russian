"""Крошечный «трансформер» для тестов aiw_ru.models: ONNX-граф с тем же входом и выходом.

Настоящая модель весит десятки мегабайт и требует torch для экспорта. Этот граф
принимает input_ids и attention_mask и отдаёт logits на два класса: logit ИИ растёт
со средним номером токена, поэтому слова из конца словаря тянут текст к ИИ.
Пересобрать (нужен пакет onnx из группы transformer):

    uv run --group transformer python tests/fixtures/tiny-transformer/make.py
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent

nodes = [
    helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT),
    helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
    helper.make_node("Mul", ["ids_f", "mask_f"], ["masked"]),
    helper.make_node("ReduceSum", ["masked", "axis1"], ["total"], keepdims=1),
    helper.make_node("ReduceSum", ["mask_f", "axis1"], ["count"], keepdims=1),
    helper.make_node("Div", ["total", "count"], ["mean"]),
    helper.make_node("Sub", ["mean", "middle"], ["ai"]),
    helper.make_node("Neg", ["ai"], ["human"]),
    helper.make_node("Concat", ["human", "ai"], ["logits"], axis=1),
]
graph = helper.make_graph(
    nodes,
    "tiny",
    [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "tokens"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "tokens"]),
    ],
    [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
    initializer=[
        helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("middle", TensorProto.FLOAT, [], [8.0]),
    ],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="aiw-ru tests")
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, HERE / "model.onnx")
