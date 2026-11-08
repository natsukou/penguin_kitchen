# 固定桌面识别与 G1 测距

## 类别和参考样本

`tabletop-fit` 直接读取 `working/frontend/app.js` 中的 12 个菜谱 ID、菜品名称和对应照片，以及蘑菇、豆腐、鸡胸肉、新鲜蔬菜 4 个食材组。菜谱 ID 继续由前端维护。识别时显式选择成品 `recipe`、原始食材 `ingredient` 或实物 `tabletop`，避免拿参考菜照判断眼前器具。实物初版包含透明水瓶、粉色水壶、碗、杯、带孔勺和盒子。

本地 YOLO11n 提取神经特征，按类拟合特征中心。使用 `--propose` 时，先在指定桌面区域检测瓶、碗、杯和勺的候选框，再在相应类别内匹配。也可用 `--roi` 对指定物体裁图独立分类。相似度和领先第二名的差值只用于筛选，分别由 `--min-similarity` 和 `--min-margin` 配置。没有来自至少两组现场采集的参考，或匹配不明确时，类 ID 为 `unknown`。两组只是进入候选的最低门槛，识别准确率仍需用独立现场数据验证。模型报告保留 YOLO 检测分数和神经特征相似度，两者含义不同。

`--references` 接收 JSON 对象，`references` 是实际照片的标注列表。每条有 `class_id`、`image`、`xyxy_px`、`capture_id`，以及可选 `kind`（默认 `tabletop`）。图片路径相对该 JSON 所在目录，框是原图像素 `[x1,y1,x2,y2]`。同批次多相机图和同一段视频的帧应共享采集 ID。给菜谱或食材加现场照片时，使用已有前端 ID 和对应的 `kind`。留出另一场采集的样本验证，不能把做过参考的图片当作验收。当前原型使用现有检测器的候选框，召回不足时需用现场标注微调检测器。

```powershell
python -m pet_meal tabletop-fit --references <现场标注.json> --output <原型库.npz>
python -m pet_meal tabletop-recognize --bank <原型库.npz> --kind tabletop --image <图片> --roi X1 Y1 X2 Y2 --propose
```

## 目标距离与桌面坐标

使用同一腕部的配对 RGB-D、逐帧相机标定和采集时刻的 G1 坐标变换。操作者在原始 RGB 图上标出同一物体的表面像素，系统检查其附近的深度一致性并反投影到 `base_link`。可再标桌面的原点、沿 X 方向的一点、沿正 Y 侧的一点，得到右手桌面坐标系。三点结果是桌面局部平面估计，后续定位跨批次需要重新观测固定桌面标记或标定变换。

报告分别给出相机到表面的直线距离、基座原点的平面距离、同时间戳的安装法兰到表面的距离。提供对应手臂及工具的标定 TCP 偏移后，才计算夹持中心到表面的距离。这里测的是表面点，抓取位姿需要额外计算，能否到达由逆运动学和包含桌面、夹爪的碰撞场景判断。

```powershell
python -m pet_meal g1-tabletop-range --capture <采集目录> --wrist right --pixel U V --table-pixels U0 V0 UX VX UY VY
```

## 夹爪扭矩接口

SDK 的开口单位为 m、速度为 m/s、`effort` 为 N·m。夹爪反馈会和采集时间做配对。读出的扭矩不能直接当作指尖接触力 N。要测量接触力，需使用实装夹爪、指垫和力计标定。

`--tool-profile` 提供现场标定结果，字段为 `side`、`flange_frame`、`calibration_id`、`tool_model`、`tcp_offset_in_flange_m`，以及 `limits.width_range_m`、`limits.max_speed_mps`、`limits.max_effort_nm`。可用 `--grip WIDTH_M SPEED_MPS EFFORT_NM` 对一条假设命令做离线检查。平台依次检查 SDK 输入范围、标定限值和配对的夹爪反馈。本入口的输出供核验，动作执行还需已验证的工具模型、碰撞规划和现场力控验收。

- [Galbot G1 SDK 1.10.0](https://developer.galbot.com/docs/SDK/1.10.0/g1/en/api_python_reference/)
- [Ultralytics 本地特征提取](https://docs.ultralytics.com/modes/predict)
- [MoveIt PlanningScene](https://moveit.picknik.ai/humble/api/html/classplanning__scene_1_1PlanningScene.html)
