# VACC4GTA: 基于导航地图的GTA自动驾驶AI

## 项目概述
本项目参考[Sandeep354/GTA5-SelfDrive](https://github.com/Sandeep354/GTA5-SelfDrive)实现基于导航地图的深度学习自动驾驶系统，专注于通过游戏小地图图像实现车辆巡线行驶。

## 核心功能
- 基于小地图图像的路径识别与导航
- 车辆方向控制（左转、直行、右转）
- 深度学习模型训练与推理框架

## 项目结构
```
VACC4GTA/
├── data/           # 训练数据与预处理结果
├── models/         # 模型定义与训练权重
├── scripts/        # 数据采集与处理脚本
├── src/            # 核心代码实现
├── requirements.txt # 项目依赖
└── README.md       # 项目文档
```

## 使用方法
1. 安装依赖: `pip install -r requirements.txt`
2. 数据采集: 运行`scripts/capture_data.py`捕获游戏小地图与控制数据
3. 数据预处理: 使用`scripts/balance_data.py`处理训练数据
4. 模型训练: 执行`src/train.py`训练自动驾驶模型
5. 推理运行: 启动`src/inference.py`实现游戏内自动驾驶