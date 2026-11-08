# 识别与决策接口 V1

执行链路以 `Observation → FixedScenePolicy → Action → SafetyGuard` 为准。当前接口在 Python 进程内使用 [数据类型](../pet_meal/types.py)，下文的 JSON 与监控日志中对应事件的 `payload` 形状一致。日志外层的 `schema_version` 为 `1`。百炼的可见性证据使用独立接口。

## 模块分工

- **视觉定位**接收 RGB-D 与标定，输出每个任务物体在机器人基座坐标系中的位置、置信度和观测来源。视觉无需知道当前任务阶段。
- **执行反馈**报告夹持对象、勺内是否有食物、碗内是否有食物及反馈来源。后端把反馈与定位结果合成一份 `Observation`。当前这些状态由模拟器提供。
- **任务决策**持有任务阶段和历史信息，例如首次观察到的勺归位位置。它决定当前操作什么，生成一个高层 `Action`，并在执行后根据新观测确认是否推进。
- **安全检查**在派发前核对观测时效、RGB 与深度的时间差、目标位置、坐标系、工作区和动作前置条件。执行器接收通过检查的动作。

## 观测输入 `Observation`

```json
{
  "timestamp_ns": 827301250000000,
  "detections": {
    "spoon": {"object_id": "spoon", "position_m": [0.38, -0.18, 0.10], "confidence": 1.0, "frame_id": "base_link"},
    "food_container": {"object_id": "food_container", "position_m": [0.60, -0.18, 0.10], "confidence": 1.0, "frame_id": "base_link"},
    "bowl": {"object_id": "bowl", "position_m": [0.42, 0.18, 0.10], "confidence": 1.0, "frame_id": "base_link"},
    "mat": {"object_id": "mat", "position_m": [0.65, 0.18, 0.07], "confidence": 1.0, "frame_id": "base_link"}
  },
  "feedback": {"held_object": null, "spoon_loaded": false, "bowl_loaded": false, "source": "simulated_state"},
  "source": "synthetic_rgbd_markers",
  "sensor_skew_ns": 0
}
```

- `timestamp_ns` 为与安全检查同一主机时钟域的 `time.monotonic_ns()` 时间，单位纳秒。日志外层的 `utc` 用于查看记录时间。观测年龄以本机单调时钟计算。
- `sensor_skew_ns` 为 RGB 与深度的时间差绝对值，单位纳秒。当前模拟默认允许观测年龄最多 1 秒、RGB 与深度时间差最多 50 毫秒。
- `detections` 按任务物体 ID 索引，每个 ID 至多一条有效定位，键与 `object_id` 必须一致。`position_m` 是 `base_link` 下三个有限数值 `[x, y, z]`，单位米。置信度范围为 `[0, 1]`，模拟默认门槛为 `0.8`，与 YOLO 的实测阈值无关。当前视觉适配器省略未检出的目标，遇到重复标记或有效深度不足会报错。完整流程要求每次观测都含四个目标，缺失或检查失败会锁存停止。
- `feedback.held_object` 为抓持对象的 ID，`null` 表示已知空夹持。`spoon_loaded` 与 `bowl_loaded` 为布尔状态。当前格式表示已知反馈，未来实机的未知状态需单独表达，不能编码成空夹持或空碗。
- 当前任务入口只接受来源 `synthetic_rgbd_markers` 和反馈来源 `simulated_state`。接入实机前需核验采集时钟、标定、目标角色映射及真实反馈来源。

## 决策输出 `Action`

```json
{
  "name": "grasp_spoon",
  "object_id": "spoon",
  "target_m": [0.38, -0.18, 0.10],
  "frame_id": "base_link"
}
```

动作与目标 ID 的组合固定：

- `grasp_spoon → spoon`
- `acquire_food → food_container`
- `pour_food → bowl`
- `return_spoon → spoon`，`target_m` 取第一次观测记下的勺归位点
- `grasp_bowl → bowl`
- `place_bowl → mat`，`object_id` 表示目的地，当前持有的物体由反馈确认为 `bowl`

除归位例外，`target_m` 取当前观测中的目标位置。这个接口只表达高层动作和目标点，抓取姿态、轨迹及夹爪指令由执行侧处理。动作提交后重新观测，分别核验抓持、取食、倒入、释放或放置结果。观察、定位确认和最终核验阶段可以推进状态但不生成动作。

## 百炼与本地 YOLO

百炼的响应格式是 `{"objects": [...]}`，四个任务角色各一项，字段只有 `role`、`visibility`、`evidence`。`visibility` 取 `visible`、`uncertain` 或 `not_visible`，`evidence` 是 1–240 字符的可见证据。任务阶段由本地相关性规则另行传入。规则输出 `phase`、`target_role`、`outcome`、`objects` 和 `dispatch_allowed`，每个物体附加 `relevance`。目标可见时为 `candidate`，否则为 `hold`，两种结果的 `dispatch_allowed` 均为 `false`。无效响应由解析器拒绝。

YOLO 当前输出 `class_id`、`label`、`confidence` 和 `box_xyxy_px`。接入执行观测前需要把类别和实例映射为任务 ID，结合对齐深度和标定计算基座坐标。二维框和百炼可见性报告都留在识别证据层，只有完成定位并通过安全检查的目标才能进入动作派发。

## 场景识别与抓取引导

`observation_all.json` 进入 [场景识别模型](../pet_meal/recognition.py)。每个检测条目代表一个实例，名称和描述记录语义证据，置信度用于解释识别结果。按实例 ID 或类别查找目标。若类别命中多个实例，保留所有候选供区分。场景时间戳可以缺失，原始位置与坐标系必须同时填写或留空。反馈保留 `"unknown"`（未知夹持）、`null`（报告为空）和实例 ID（报告已持有），勺、碗状态的 `null` 表示未知。

对同一目标做定位时，给该检测补充一个 `landmark` 对象。其 `name` 是同一物理特征的名称，例如 `cap_center`。`views` 以采集的相机 ID 为键，值是 `{"timestamp_ns": 采集帧时间戳, "pixel_px": [u, v]}`。要求包含 `head-left-rgb`、`head-right-rgb` 和至少一路 `left-wrist-rgb` 或 `right-wrist-rgb`，像素和时间戳均来自对应原始分辨率的采集帧。当前由上游或人工提供同名特征的像素关联。YOLO 边框中心无法替代跨视角的同一物理特征。

[G1 多相机定位](../pet_meal/g1_multiview.py) 读取同批 `capture.json` 和腕部原始深度，核对像素引用与帧时间戳、参与帧的 50 毫秒窗口和逐帧坐标变换。头部双目三角化的 `base_link` 特征点，与各路已提供腕部的深度反投影逐一比较。三厘米是当前几何交叉核验门槛。源文件的 `position_m` 只作为报告字段保留。单个特征点也无法表示夹持姿态。

动作策略按证据选择下一步。目标缺失时输出 `reobserve_target`，多个实例时输出 `disambiguate_target`，缺少或拒绝定位证据时输出 `localize_target`。通过定位核验后，夹持未知或已占用时输出 `verify_gripper`，报告空夹持时输出 `plan_grasp`。最后一步表示可以交给规划层计算夹持姿态与轨迹。这一离线入口的 `dispatch_allowed` 始终为 `false`，实机执行仍需新鲜的观测、真实夹爪反馈和碰撞检查。

```powershell
.\.venv\Scripts\python.exe -m pet_meal scene-action --observation observation_all.json --object-id bottle_01
```

可用 `--category bottle` 检查类别歧义。提供带 `landmark` 的观测后，用 `--capture <采集目录>` 核验三维位置，可加 `--output <报告路径>` 保存结果。

固定菜谱、实物参考识别和 G1 的腕部距离与夹爪扭矩检查见 [固定桌面识别与 G1 测距](tabletop-platform.md)。
