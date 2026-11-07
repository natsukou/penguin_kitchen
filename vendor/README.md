# ACT 与数据转换工具

源码包保留上游许可证、README、构建配置和完整 Python 包源码。以下 SHA-256 可用于校验仓库中的归档文件。

- `lerobot-0.3.3.tar.gz` 取自 `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`，含 ACT 的模型、配置和训练入口。SHA-256 为 `61a7efe4f1fbb1a61998d0264f9d0754818958aa663bc1a881bffa8326534683`。[LeRobot 上游](https://github.com/huggingface/lerobot/tree/b883328e6c95681ca90a18b102e4ae5e1f91e2bf)
- `galbot-mcap2lerobot-b8d7dbb.tar.gz` 取自 `b8d7dbbe494c054a9427416238309c59b6a428ce`，用于转换 Galbot MCAP 为 LeRobotDataset v2.1。SHA-256 为 `2ad60378a21fd87ff7f49882cc1ffbba1c4e3c617ccce557e250a137574bf775`。[转换器上游](https://github.com/GalaxyGeneralRobotics/galbot-mcap2lerobot/tree/b8d7dbbe494c054a9427416238309c59b6a428ce)

在仓库根目录打开 PowerShell，用 Python 3.11 创建独立环境。pip 仍需获取 PyTorch 等运行依赖。转换器的 FFmpeg 安装和录制要求见其上游 README。

```powershell
python -m venv .venv-act
.\.venv-act\Scripts\python.exe -m pip install ./vendor/lerobot-0.3.3.tar.gz ./vendor/galbot-mcap2lerobot-b8d7dbb.tar.gz
```

两个源码包已验证可构建 wheel。实际训练需要与 G1 相机、状态和动作维度匹配的示范数据，本项目目前没有训练集或 ACT 检查点。

随仓库分发的公开餐具照片由 SomeBodyAnyBody05 发布，采用 CC0 1.0。[图片来源与许可](https://commons.wikimedia.org/wiki/File:Fiber_One_in_a_bowl_with_milk_and_a_spoon.jpg)
