## 整体流程

### 1.流程阶段和平台工具

|   **阶段**   |          **平台**           |            **核心工具**             |         **作用**          |
| :----------: | :-------------------------: | :---------------------------------: | :-----------------------: |
| **模型导出** | **Windows/Linux PC（x86）** |         **PyTorch + ONNX**          | **将`.pth`转为中间格式**  |
| **模型转换** |     **Orin设备（ARM）**     | **TensorRT（Python API或trtexec）** | **生成硬件专用`.engine`** |
| **部署推理** |     **Orin设备（ARM）**     | **TensorRT Runtime + CUDA/PyCUDA**  |   **加载引擎执行推理**    |

### 2.在windows下实现.pth转.onnx

- #### 安装依赖

```cmd
pip install onnx onnxscript
```

- #### 输出onnx模型

```python
import torch
from model import YourModel  # 你的网络定义

# 加载模型
model = YourModel() 
model.load_state_dict(torch.load('model.pth')) # 加载权重
model.eval() # 设置推理模式

# 构造输入
dummy_input = torch.randn(1, 3, 224, 224) # 形状为 (batch_size, channels, height, width)

# 导出ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes=None,
    opset_version=11
    dynamo=False
)
```

|   常见参数    |              含义               |
| :-----------: | :-----------------------------: |
|     model     |     需要导出的 PyTorch 模型     |
|  dummy_input  |          模型输入样例           |
|   onnx_path   |          ONNX 保存路径          |
|  input_names  |         给输入节点命名          |
| output_names  |          输出节点名字           |
| opset_version |          ONNX 算子版本          |
| dynamic_axes  |        设置动态输入尺寸         |
|    dynamo     | PyTorch >=2.1后dynamo默认为True |

### 3.在Orin上实现.onnx转.engine

- #### trtexec 命令行

```cmd
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16 \  # 开启半精度，Orin FP16性能翻倍
        --workspace=1024 \
        --minShapes=input:1x3x224x224 \
        --optShapes=input:8x3x224x224 \
        --maxShapes=input:16x3x224x224  # 若需动态batch
```

- #### Python API

```python
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 解析ONNX
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 构建配置
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # 开启FP16

# 生成引擎
engine = builder.build_engine(network, config)
with open("model.engine", "wb") as f:
    f.write(engine.serialize())
```

### 4.在Orin上加载推理.engine

```python
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt

# 加载engine
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open("model.engine", "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# 准备输入（固定尺寸）
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
input_buffer = cuda.mem_alloc(input_data.nbytes)
output_buffer = cuda.mem_alloc(1 * 1000 * 4)  # 假设输出1000类

# 执行推理
cuda.memcpy_htod(input_buffer, input_data)
context.execute_v2([int(input_buffer), int(output_buffer)])
cuda.memcpy_dtoh(output_data, output_buffer)
```

### 5.性能和精度测试

- #### 性能测试（trtexec）

```cmd
# 测延迟（纯推理耗时）
trtexec --loadEngine=model.engine \
        --iterations=100 \
        --warmUp=10 \
        --avgRuns=50

# 测吞吐量（QPS，适合批量场景）
trtexec --loadEngine=model.engine \
        --iterations=100 \
        --streams=4 \  # 4个并行流
        --useCudaGraph  # 启用CUDAGraph减少调度开销
```

- #### 精度测试（trtexec）

```cmd
# 需要同一份输入数据（生成随机或真实数据）
trtexec --loadEngine=model.engine \
        --onnx=model.onnx \
        --iterations=100 \
        --dumpOutputs \
        --dumpRawBindings \
```

