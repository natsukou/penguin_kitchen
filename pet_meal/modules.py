"""Module inventory and last observed health, not fabricated device heartbeats."""

from copy import deepcopy

MODULES = (
    ("vision", "视觉感知", "RGB-D 标记检测 / 基座坐标定位"),
    ("decision", "任务编排", "有限状态机 / 阶段转换"),
    ("policy", "动作策略", "固定场景确定性策略"),
    ("safety", "控制检查", "观测时效 / 工作区 / 前置条件"),
    ("control", "执行适配", "模拟器命令 / 单一执行入口"),
    ("feedback", "结果确认", "抓持 / 食物状态 / 放置反馈"),
    ("recorder", "运行记录", "JSONL / 图像 / 源码与配置哈希"),
)


class ModuleRegistry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.modules = {
            key: {
                "id": key,
                "name": name,
                "description": desc,
                "status": "ready",
                "last_event_utc": None,
                "detail": "等待运行",
            }
            for key, name, desc in MODULES
        }
        self.modules["hardware"] = {
            "id": "hardware",
            "name": "Galbot G1",
            "description": "实机 SDK / 控制器",
            "status": "disabled",
            "last_event_utc": None,
            "detail": "等待 HPU/XCU 回执",
        }
        self.modules["act"] = {
            "id": "act",
            "name": "ACT 学习策略",
            "description": "条件实验分支",
            "status": "disabled",
            "last_event_utc": None,
            "detail": "未训练 / 未加载",
        }

    def consume(self, row: dict):
        name, payload = row["event"], row.get("payload", {})
        owners = {
            "state_entered": ["decision"],
            "observation": ["vision"],
            "action_selected": ["policy"],
            "action_checked": ["safety"],
            "action_dispatched": ["control"],
            "action_confirmed": ["feedback"],
            "task_target_resolved": ["vision", "policy"],
            "task_candidate": ["decision"],
            "task_stage_reviewed": ["safety"],
            "hpu_stage_checked": ["hardware", "safety"],
            "hpu_sdk_receipt": ["hardware", "control"],
            "hpu_stage_verified": ["hardware", "feedback"],
            "run_started": ["recorder"],
            "run_finished": ["recorder"],
        }
        for owner in owners.get(name, []):
            self.modules[owner].update(status="healthy", last_event_utc=row["utc"], detail=name)
        if name == "failure":
            owner = payload.get("module", "decision")
            if owner in self.modules:
                self.modules[owner].update(
                    status="fault", last_event_utc=row["utc"], detail=payload["error"]
                )

    def snapshot(self) -> list[dict]:
        return deepcopy(list(self.modules.values()))
