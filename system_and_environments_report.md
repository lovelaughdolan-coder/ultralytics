# 工作站硬件配置与虚拟环境报告
*最后更新时间：2026-04-09*

## 1. 硬件配置 (Hardware Specifications)

| 项目 | 详细规格 |
| :--- | :--- |
| **操作系统** | Ubuntu 22.04.5 LTS |
| **CPU** | Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz (核心/线程详见 lscpu) |
| **GPU** | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| **内存 (RAM)** | 64GiB (可用 ~47GiB) |
| **驱动版本** | NVIDIA Driver 590.48.01 (支持至 CUDA 12.8+) |

## 2. AI 环境兼容性 (PyTorch & CUDA)

为了保证深度学习任务的稳定性，以下是本机验证通过的软件组合：

| 环境名称 | PyTorch 核心 | CUDA / 关键组件版本 | 备注 |
| :--- | :--- | :--- | :--- |
| **animal-3d** | 2.10.0 | CUDA 12.8 / torchvision 0.26.0.dev | **[已弃用]** 仅供历史参考 |
| **blenderproc** | 2.10.0 | CUDA 12.8 / torchaudio 2.3.1+3edcf69 | 用于合成数据测试 |
| **yolo** | 2.10.0 | CUDA 12.8 / torchvision 0.18.1 | 相对稳定的模型推理组合 |
| **roboflow** | 2.10.0 | CUDA 12.8 | 基础推理环境 |

## 3. 虚拟环境概览 (Micromamba Environments)

以下是当前系统维护的虚拟环境及其核心用途说明：

| 环境名称 | 核心用途 | 主要依赖包 (核心) |
| :--- | :--- | :--- |
| **animal-3d** | **[已弃用]** 历史动物重建项目环境，目前已不再使用。 | `torch` 等 |
| **blenderproc** | **合成数据测试**。用于测试 BlenderProc 自动化渲染和合成数据集。 | `blenderproc`, `bpy` |
| **yolo** | **YOLO 专用推理/训练**。用于 Ultralytics 系列模型的开发与测试。 | `ultralytics`, `torch`, `opencv-python` |
| **roboflow** | **数据集管理与模型推理**。用于对接 Roboflow 平台的数据处理与模型测试。 | `roboflow`, `inference-sdk`, `opencv-python` |
| **mobile-sam** | **轻量化分割模型**。专门用于运行和测试 Mobile-SAM 分割算法。 | `mobile-sam`, `torch`, `segment-anything` |
| **mmpose_stable**| **姿态估计备份**。稳定的 MMPose 环境，用于对比或作为备选姿态估计方案。 | `mmpose`, `mmcv-full`, `torch` |
| **yolov5** | **旧版模型兼容**。用于兼容和维护运行基于 YOLOv5 的旧项目代码。 | `yolov5` 相关依赖, `torch` |
| **base** | 基础管理环境。 | micromamba 核心组件 |

## 4. 源码构建与本地仓库 (Source Build & Local Repos)

部分环境包含从源码编译的重度依赖，以下是历史构建中的关键信息：

### 4.1 辅助/测试代码库
这些目录通常用于测试开源视觉项目或作为非重点项目的支撑：
- **BlenderProc**: `/home/hxy/blenderproc` (环境 `blenderproc`)
- **MobileSAM**: `/home/hxy/MobileSAM`
- **YOLOv5**: `/home/hxy/yolov5`

### 4.2 机器人/控制核心库 (Current Focus)
- **MoveIt/Robot SDK**: `/home/hxy/elfin_SDK`, `/home/hxy/ros2_ws`

### 4.2 历史构建避坑指南 (Build Notes)
在恢复或重建 `animal-3d` 或 `mmpose` 相关环境时，请参考以下编译要点：
- **CUDA 算子编译**：
  - `nvdiffrast`: 需通过 `pip install git+https://github.com/NVlabs/nvdiffrast/` 源码安装。
  - `tiny-cuda-nn`: 必须指定子目录编译：`pip install git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch`。
- **OpenMMLab 体系**：
  - `mmpose` 依赖 `mmcv` 和 `mmengine`。若 GPU 驱动更新，可能需要重新编译 `mmcv` 以匹配最新的 CUDA 版本。
- **构建环境**：编译时需要确保 `gcc` 版本与 `nvcc` 兼容（本机已知 `gcc 11.4.0` 配合 `cuda 12.8` 运行正常）。

## 5. 快速上手建议 (Quick Start for New Agents)

- **图像检测/分割**：优先考虑 `yolo` 或 `roboflow` 环境。
- **3D 动物/机器人开发**：必须切换至 `animal-3d`，它集成了 ROS 2 骨干和 AI 框架。
- **渲染/合成数据**：使用 `blenderproc`，注意该环境已挂载本地工作空间。
- **环境切换命令**：`micromamba activate <env_name>`
