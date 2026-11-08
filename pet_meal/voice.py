"""Interpret explicit Chinese tabletop commands without scheduling a run."""

import re
import unicodedata

from .g1_task import TaskRequest
from .types import TaskError

PICK = r"(?:抓取|抓起|抓|拿起|拿住|取)(?:矿泉水瓶|塑料水瓶|水瓶|瓶子)"
RETURN = r"(?:再|后|然后)?放回(?:原位|原处)?"
SHAKE = r"(?:再|后|然后)?摇(?:动)?([一二两三123])(?:下|次)"
COUNTS = {"一": 1, "二": 2, "两": 2, "三": 3, "1": 1, "2": 2, "3": 3}


def interpret_voice(text: str) -> TaskRequest:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 120:
        raise ValueError("请输入 1 至 120 个字符的转写文本")
    command = unicodedata.normalize("NFKC", text)
    command = re.sub(r"[\s,，]", "", command).rstrip("。.!！?？")
    command = re.sub(r"^(?:请帮我|帮我|请)", "", command, count=1)

    if re.fullmatch(PICK, command):
        return TaskRequest("grasp", category="bottle")
    if re.fullmatch(PICK + RETURN, command):
        return TaskRequest("pick_place", category="bottle")
    shake = re.fullmatch(PICK + SHAKE + RETURN, command)
    if shake:
        return TaskRequest("shake_then_place", category="bottle", repeat_count=COUNTS[shake[1]])
    raise TaskError("指令未确认。请说「抓水瓶」「取水瓶后放回」或「抓起水瓶摇一下再放回」。")
