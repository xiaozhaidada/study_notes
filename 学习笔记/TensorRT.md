## TensorRT

### 1.确定版本兼容性（https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html）

- #### TensorRT version

- #### Platform

- #### CUDA Toolkit

- #### cuDNN

- #### ONNX

- #### Python

### 2.准备环境（https://bqcode.blog.csdn.net/article/details/135853106）

- #### NVIDIA GPU 驱动

  ```cmd
  # 打开 CMD / PowerShell
  nvidia-smi
  
  # 正常输出
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 560.94       Driver Version: 560.94       CUDA Version: 12.6     |
  |-------------------------------+----------------------+----------------------+
  | GPU Name                      | Memory-Usage         |
  | NVIDIA GeForce RTX 4090       | ...
  +-----------------------------------------------------------------------------+
  ```

- #### CUDA（[CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive)）

  ```cmd
  # 打开 CMD / PowerShell
  nvcc -V
  
  # 正常输出
  nvcc: NVIDIA (R) Cuda compiler driver
  Copyright (c) 2005-2026 NVIDIA Corporation
  Built on Tue_Jun__9_14:30:19_Pacific_Daylight_Time_2026
  Cuda compilation tools, release 12.6, V12.6.3  // 要与 CUDA Version 匹配
  Build cuda_12.6.r12.6mpiler.38244171_0
  ```

- #### cuDNN（[cuDNN Archive | NVIDIA Developer](https://developer.nvidia.com/rdp/cudnn-archive)）

  ```cmd
  # 打开 CMD / PowerShell
  dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" // 先查看装目录，获取 CUDA 版本
  dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudnn*.dll" // v12.6 是 CUDA 版本，请根据上面查询的结果修改版本
  
  # 正常输出
  2023/11/30  16:26           288,296 cudnn64_8.dll // 不一定是这个版本
  2023/11/30  16:26       125,217,320 cudnn_adv_infer64_8.dll
  2023/11/30  16:26       116,558,888 cudnn_adv_train64_8.dll
  2023/11/30  16:26       582,690,344 cudnn_cnn_infer64_8.dll
  2023/11/30  16:26       122,242,104 cudnn_cnn_train64_8.dll
  2023/11/30  16:26        89,759,272 cudnn_ops_infer64_8.dll
  2023/11/30  16:26        70,162,472 cudnn_ops_train64_8.dll 
  ```

- #### TensorRT

  ```cmd
  pip install tensorrt-cu12 // tensorrt-cu 版本根据你的 CUDA 版本
  ```

  
