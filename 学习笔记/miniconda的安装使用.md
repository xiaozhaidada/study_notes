## 环境配置

### 1.conda环境

#### 下载 Miniconda 安装包

```bash
# x86_64电脑：
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# ARM64设备：
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
///
	如果下载的太慢，可以换源，也可以先下载到window中再通过共享文件传给Ubuntu
///
```

#### 安装 Miniconda

```bash
# x86_64电脑：
bash Miniconda3-latest-Linux-x86_64.sh
# ARM64设备：
bash Miniconda3-latest-Linux-aarch64.sh
```

#### 初始化 conda

```bash
~/miniconda3/bin/conda init bash
```

#### 激活 conda

```bash
source ~/.bashrc
```

#### 创建 python 环境

```bash
# 创建环境
conda create -n 环境名 python=版本号
# 激活环境
conda activate 环境名
# 退出环境
conda deactivate
```

