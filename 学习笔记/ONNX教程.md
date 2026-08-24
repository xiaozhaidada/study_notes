



## ONNX教程（[Export a PyTorch model to ONNX — PyTorch Tutorials 2.13.0+cu130 documentation](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html?source=post_page-----9c7397366904---------------------------------------&utm_source=chatgpt.com)）

#### 1.安装依赖

```cmd
pip install --upgrade onnx onnxscript
```

#### 2.构建模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### 3.输出onnx模型

```python
torch_model = ImageClassifierModel()
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 1, 32, 32),)
onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
```

```python
# torch.onnx.export常见默认参数
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=None,
    output_names=None,
    opset_version=11,
    dynamic_axes=None,
    dynamo=False
)
```

|     参数      |              含义               |
| :-----------: | :-----------------------------: |
|     model     |     需要导出的 PyTorch 模型     |
|  dummy_input  |          模型输入样例           |
|   onnx_path   |          ONNX 保存路径          |
|  input_names  |         给输入节点命名          |
| output_names  |          输出节点名字           |
| opset_version |          ONNX 算子版本          |
| dynamic_axes  |        设置动态输入尺寸         |
|    dynamo     | PyTorch >=2.1后dynamo默认为True |

```python
# ps:
# PyTorch <= 2.5 在torch.onnx.export参数中可以保存模型，如下所示
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx"
)
# PyTorch >= 2.6 需要调用.save()保存模型
```

#### 4.保存和加载onnx模型

```python
# 保存
onnx_program.save("image_classifier_model.onnx")


# 加载
import onnx

onnx_model = onnx.load("image_classifier_model.onnx")
onnx.checker.check_model(onnx_model)
```

#### 5.执行模型

```python
# pip install onnxruntime

import onnxruntime

onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
print(f"Input length: {len(onnx_inputs)}")
print(f"Sample input: {onnx_inputs}")

ort_session = onnxruntime.InferenceSession(
    "./image_classifier_model.onnx", providers=["CPUExecutionProvider"]
)

onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# ONNX Runtime returns a list of outputs
onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]
```

