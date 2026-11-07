# Pet Meal Robotics

银河通用 Galbot G1 宠物餐配置演示项目，V0.2，2026-11-06。

## 本地控制中枢

```powershell
.\.venv\Scripts\python.exe -m pet_meal monitor
```

在浏览器打开 [本地控制台](http://127.0.0.1:8765)，选择“正常完整流程”并启动。可以在运行中暂停、继续或软件停止，也可以选择故障场景观察停止原因。故障和软件停止都会锁存，复位后可重新启动。终端按 `Ctrl+C` 结束服务。端口占用时可以传入 `--port` 使用其他本机端口。

界面包含任务进度、目标定位、合成相机帧、执行反馈、模块状态、测试报告和运行历史。每次运行的图像、JSONL 和摘要存放在 `artifacts/hub-runs/`。模块职责、命令状态与测试边界见 [控制中枢设计](docs/control-center.md)。

真实照片推理结果保存在 `artifacts/vision-tableware/`，可在控制台的测试报告中查看。这项测试单独运行，完整任务使用合成场景。

百炼视觉适配器支持用同一份云端观测比较不同阶段的目标相关性。运行 `python -m pet_meal relevance-test` 可生成离线规则报告。密钥与实测命令见 [百炼视觉接入](docs/bailian-vision.md)。

当前交付是**可运行的离线软件基线**：合成 RGB-D 标记检测、相机到基座定位、有限状态机、确定性动作选择、控制入口检查、执行后反馈、JSONL 日志与故障测试。已经下载真实 YOLO11n 预训练权重，并通过本机 GPU 推理测试。

**没有连接机器人，没有完成实机验收，没有训练 ACT。** 合成环境不模拟机器人关节、碰撞、接触力或食品流动，运行耗时不代表 2–3 分钟实机演示时长。实机动作接口在代码中禁用，仅修改配置无法启用。

## 快速运行

在仓库根目录打开 PowerShell。首次克隆后运行 `.\scripts\bootstrap.ps1` 创建开发环境。软件测试可断网运行，HTTP 集成测试需要本机回环连接。

```powershell
.\.venv\Scripts\python.exe -m pet_meal doctor
.\.venv\Scripts\python.exe -m pet_meal simulate
.\.venv\Scripts\python.exe scripts/validate.py
```

故障演示返回非零退出码，属于预期结果：

```powershell
.\.venv\Scripts\python.exe -m pet_meal simulate --fault grasp_slip
.\.venv\Scripts\python.exe -m pet_meal simulate --fault placement_error
```

实际模型推理（输出目录必须尚不存在）：

```powershell
.\.venv\Scripts\python.exe -m pet_meal detect-image --weights models/yolo11n.pt --image models/bus.jpg --device 0 --output artifacts/yolo-recheck
```

GPU 不可用时用 `--device cpu`。官方 `bus.jpg` 只检验推理入口，不用于证明宠物餐物体识别准确率。YOLO 当前与任务控制隔离，检测框不能直接作为抓取位姿。

## 已下载资源

- `external/GalbotSDK`：官方 SDK 1.10.0、Linux 动态库、接口文档和示例。
- `external/galbot_one_golf_description`：官方 URDF、MJCF、USD 模型。
- `external/galbot-mcap2lerobot`：官方 MCAP → LeRobotDataset v2.1 转换器。
- `external/lerobot`：官方 v0.3.3 源码，与上述转换器的依赖声明匹配；尚未安装完整训练环境。
- `models/yolo11n.pt`：官方 COCO 预训练模型；来源、体积及 SHA256 见 `models/assets.lock.json`。

提交号见 `external.lock.json`。下载脚本还会获取并校验百炼测试使用的公开餐具照片。首次克隆后按需要获取外部资源：

```powershell
.\scripts\fetch_official.ps1
.\scripts\download_vision_assets.ps1
```

首次建立 Python 环境可运行 `scripts/bootstrap.ps1`。当前 `.venv` 使用 `--system-site-packages` 复用本机 PyTorch/CUDA，环境信息见 `artifacts/environment.json`；这不是完全独立的环境。完整机器学习训练环境应另建环境，不能认为已下载源码等于已验证全部依赖。

## 软件入口与输出

- `pet_meal/vision.py`：固定颜色标记分割、深度有效性检查、三维反投影与坐标变换。
- `pet_meal/decision.py`、`policy.py`：任务状态机与根据新观测选择动作的固定场景策略。
- `pet_meal/safety.py`：观测新鲜度、传感器时间差、身份/坐标系、工作区、动作前置条件、停止锁存。
- `pet_meal/simulator.py`：合成图像与状态反馈，包含 7 类故障注入。
- `pet_meal/sdk.py`：本地 SDK 检查、待现场验证的 Linux 只读采集入口；实机动作始终拒绝。
- `pet_meal/data.py`：完整 Episode 划分，同一采集会话不跨集合，拒绝将合成日志混入实机训练。
- `artifacts/runs/<run>/`：`events.jsonl`、`summary.json`、逐次合成 RGB 图与米制深度数组。
- `artifacts/validation.json`、`pytest-results.xml`：最新软件验证结果。
- `artifacts/yolo-smoke/`：实际 YOLO 推理结果和标注图。

模拟配置中的空间范围与阈值是**软件测试参数**，没有硬件认证含义。食品操作只表示模拟“一次取食”，不声称克重或体积定量。

更多说明见 [架构](docs/architecture.md)、[实机接入条件](docs/hardware-handoff.md)、[开发记录](docs/devlog-2026-11-06-001.md)。
