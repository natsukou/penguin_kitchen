# take01 ACT 离线测试

`act_take01_scale_23d.ckpt` 是团队为“将物体放到电子秤上”训练的 checkpoint。
测试入口 `scripts/test_act_capture.py` 复用本地 LeRobot v0.3.3 的 ACT 组件，并按
checkpoint 的 [原版 ACT 提交](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488)
保留相机特征拼接和解码层选择。所有权重按名称和尺寸严格加载。四路图像按 checkpoint
中的顺序读取，使用原版 ACT 的 RGB 归一化；离线图像缩放采用面积插值，原训练脚本的
缩放实现未在工作区找到。关节顺序和夹爪单位复用 Galbot MCAP 转换器，位置输出用
checkpoint 的统计量反归一化。

在仓库根目录使用现有 `.venv`、`external` 检出和 `artifacts/act-checkpoints/runtime`
中的运行库复测：

```powershell
$env:PYTHONPATH = 'external\lerobot\src;external\galbot-mcap2lerobot\src;artifacts\act-checkpoints\runtime'
.\.venv\Scripts\python.exe scripts\test_act_capture.py `
  --checkpoint act_take01_scale_23d.ckpt `
  --capture artifacts\g1-bowl-scale-20261108T015118.015246Z `
  --capture artifacts\g1-tabletop-startup-20261108T004837.646462Z `
  --device cuda --kinematics --output artifacts\act-take01-test\recheck.json
```

`--capture` 可重复传入。没有 GPU 时用 `--device cpu`；没有 MuJoCo 时省略
`--kinematics`。输出文件必须尚不存在。报告包含每份快照的时间差、推理耗时、输入扰动
检查和 100 步原始位置目标。角度使用弧度，夹爪开口使用米。

报告中的推理通过表示：输入快照满足 50 ms 的对齐窗口、输出尺寸和有限值检查通过、
同一输入重复输出一致，且替换图像或关节状态后输出有变化。参考 URDF 限位、负的夹爪
开口和起始目标跳变单独记录。MuJoCo 只按关节位置计算法兰坐标，并用当前关节状态对应的
法兰坐标与采集数据对照。测试没有接入机器人或运行接触动力学，报告不提供放置成功率。

本次在电子秤和另一份桌面快照上各重复推理 10 次。两组都得到有限的 `1 × 100 × 23`
输出，同设备重复输出一致；CPU 与 CUDA 的位置目标最大差小于 `4e-7`（关节弧度、
夹爪米）。RTX 3080 Laptop 的单次前向中位耗时为 25–27 ms，CPU 为 93–100 ms，
均不包含读取数据、图像预处理和轨迹验证。

电子秤快照中，`right_arm_joint5` 首步跳变为 2.149 rad，左臂最大首步跳变为
1.311 rad。MuJoCo 的起点法兰位置与现场采集值相差约 1 cm；以同一个模型投影首步
目标，右臂法兰位移约 70 cm。右夹爪 100 步中有 71 步预测负开口，最小为
−15.65 mm。另一份桌面快照同样出现首步跳变和负开口。两组输出中的 `head_joint2`
还有轻微越过参考 URDF 下限的情况（约 0.00015 rad 以内）。

原始输出暂不适合直接接入运动控制。应先用未参与训练的四路视频和 23 维轨迹核对
训练时的预处理和动作误差，再验证从实际起点进入首步的连续性、夹爪开口范围、
关节约束和碰撞。checkpoint 内的误差只来自单条训练轨迹的重建。
