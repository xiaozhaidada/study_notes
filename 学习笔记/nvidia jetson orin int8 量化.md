## nvidia jetson orin int8 量化

### 1. TensorRT PTQ

- #### 阶段

```
onnx 文件 -----> int8 量化 -----> engine/trt 文件
```

- ### 环境

```cmd
# TensorRT 版本要求
TensorRT <= 10.x
# 开发板
nvidia jetson orin
# 依赖库
pip3 install numpy pillow pycuda
```

- ### int8 量化

```cmd
# 文件结构
calibration/
  001.jpg
  002.jpg
  ...
```

```python
import os
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# 定义校准器
class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, cache_file="calibration.cache"):
        super().__init__()

        self.files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".png"))
        ]

        self.index = 0
        self.cache_file = cache_file

        self.host = np.zeros((1, 3, 224, 224), dtype=np.float32)
        self.device = cuda.mem_alloc(self.host.nbytes)

    def get_batch_size(self):
        return 1

    def preprocess(self, path):
        img = Image.open(path).convert("RGB").resize((224, 224))
        img = np.asarray(img, dtype=np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)

        return img.astype(np.float32)

    def get_batch(self, names):
        if self.index >= len(self.files):
            return None

        self.host[0] = self.preprocess(self.files[self.index])
        self.index += 1

        cuda.memcpy_htod(self.device, self.host)

        print(f"{self.index}/{len(self.files)}")

        return [int(self.device)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
```

```python
logger = trt.Logger(trt.Logger.INFO)

builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)

with open("resnet50.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()

# 开启 INT8
config.set_flag(trt.BuilderFlag.INT8)

# 校准器
config.int8_calibrator = Calibrator("calibration")

# 构建 Engine
engine = builder.build_serialized_network(network, config)

with open("resnet50_int8.engine", "wb") as f:
    f.write(engine)

print("INT8 Engine 生成完成")
```

## 2. TensorRT Model Optimizer PTQ

- ### 阶段

```
onnx 文件 -----> int8 量化 -----> onnx 文件 
```

- ### 环境

```cmd
# TensorRT 版本要求
TensorRT >= 11.x
# 开发板
nvidia jetson orin
# 依赖库
pip install --extra-index-url https://pypi.nvidia.com "nvidia-modelopt[all]"
```

- ### int8 量化

```cmd
python -m modelopt.onnx.quantization \
    --onnx_path resnet50.onnx \
    --quantize_mode int8 \
    --calibration_data calibration_data.npz \
    --output_path resnet50_int8.onnx
```

