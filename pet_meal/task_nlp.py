"""Local Chinese BERT intent and BIO slot heads for the tabletop task entry."""

import json
import unicodedata
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, BertModel

from .g1_task import SKILLS, TaskRequest
from .recognition import load_scene
from .types import TaskError

INTENTS = (*SKILLS, "serve_recipe", "hold")
SLOTS = ("category", "attribute", "destination", "repeat_count", "recipe")
TAGS = ("O", *(f"{prefix}-{slot}" for slot in SLOTS for prefix in ("B", "I")))
MAX_TOKENS = 128


class TaskHeads(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.intent = nn.Linear(size, len(INTENTS))
        self.slots = nn.Linear(size, len(TAGS))

    def forward(self, tokens):
        return self.intent(tokens[:, 0]), self.slots(tokens)


def _encoder(source):
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise TaskError("Task slots require a tokenizer with character offsets")
    return tokenizer, BertModel.from_pretrained(source, local_files_only=True).eval()


def _encode(texts, tokenizer, encoder):
    batch = tokenizer(texts, padding=True, return_offsets_mapping=True, return_tensors="pt")
    offsets = batch.pop("offset_mapping").tolist()
    if batch.input_ids.shape[1] > MAX_TOKENS:
        raise TaskError("Task text exceeds the 128-token input limit")
    with torch.no_grad():
        hidden = encoder(**batch).last_hidden_state
    return hidden, offsets


def _examples(path):
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows:
        raise TaskError("Provide labeled examples in JSONL format")
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"group", "text", "intent", "slots"}
            or not all(isinstance(row[key], str) and row[key].strip() for key in ("group", "text"))
            or len(row["text"]) > 400
            or row["intent"] not in INTENTS
            or not isinstance(row["slots"], list)
        ):
            raise TaskError("Invalid labeled task example")
    return rows


def _labels(row, offsets):
    labels = [-100 if start == end else 0 for start, end in offsets]
    for span in row["slots"]:
        name, start, end = span["label"], span["start"], span["end"]
        if (
            name not in SLOTS
            or type(start) is not int
            or type(end) is not int
            or not (0 <= start < end <= len(row["text"]))
        ):
            raise TaskError("Invalid slot span")
        indices = [i for i, (a, b) in enumerate(offsets) if a < b and a < end and start < b]
        if (
            not indices
            or offsets[indices[0]][0] != start
            or offsets[indices[-1]][1] != end
            or any(labels[i] != 0 for i in indices)
        ):
            raise TaskError("Slot spans must align with token boundaries and not overlap")
        for index, token in enumerate(indices):
            labels[token] = TAGS.index(f"{'B' if index == 0 else 'I'}-{name}")
    return labels


def fit_heads(
    train_path: Path, validation_path: Path, output: Path, source: str, epochs: int
) -> dict:
    """Fit two small heads over a frozen encoder, with separate paraphrase groups for validation."""
    train, validation = _examples(train_path), _examples(validation_path)
    if {row["group"] for row in train} & {row["group"] for row in validation} or {
        _normalized(row["text"]) for row in train
    } & {_normalized(row["text"]) for row in validation}:
        raise TaskError("Training and validation must use separate groups and texts")
    if not 1 <= epochs <= 100:
        raise TaskError("Epochs must be between 1 and 100")
    if set(INTENTS) != {row["intent"] for row in train}:
        raise TaskError("Supply training examples for every supported intent, including hold")
    source = str(Path(source).resolve()) if Path(source).is_dir() else source
    tokenizer, encoder = _encoder(source)
    torch.manual_seed(0)
    heads = TaskHeads(encoder.config.hidden_size)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=0.003)
    loss = nn.CrossEntropyLoss()

    def batches(rows):
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            hidden, offsets = _encode([row["text"] for row in batch], tokenizer, encoder)
            intents = torch.tensor([INTENTS.index(row["intent"]) for row in batch])
            slots = torch.tensor(
                [_labels(row, off) for row, off in zip(batch, offsets, strict=True)]
            )
            yield hidden, intents, slots

    train_batches = list(batches(train))
    for _ in range(epochs):
        for hidden, intents, slots in train_batches:
            optimizer.zero_grad()
            intent_logits, slot_logits = heads(hidden)
            (loss(intent_logits, intents) + loss(slot_logits.transpose(1, 2), slots)).backward()
            optimizer.step()
    heads.eval()
    correct_slots = 0
    confusion = torch.zeros((len(INTENTS), len(INTENTS)), dtype=torch.int64)
    with torch.no_grad():
        for hidden, intents, slots in batches(validation):
            intent_logits, slot_logits = heads(hidden)
            indices = intents * len(INTENTS) + intent_logits.argmax(1)
            confusion += torch.bincount(indices, minlength=len(INTENTS) ** 2).reshape_as(confusion)
            correct_slots += (
                ((slot_logits.argmax(2) == slots) | (slots == -100)).all(1).sum().item()
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        torch.save({"schema_version": 1, "encoder": source, "heads": heads.state_dict()}, stream)
    totals = confusion.sum(0) + confusion.sum(1)
    seen = totals > 0
    return {
        "train_examples": len(train),
        "validation_examples": len(validation),
        "intent_accuracy": confusion.diag().sum().item() / len(validation),
        "intent_macro_f1": (2 * confusion.diag()[seen] / totals[seen]).mean().item(),
        "slot_exact_match": correct_slots / len(validation),
        "checkpoint": str(output),
    }


def _normalized(text):
    return unicodedata.normalize("NFKC", text).strip().casefold()


def predict_text(text: str, checkpoint: Path) -> dict:
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 400:
        raise TaskError("Supply a nonempty task request of at most 400 characters")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("schema_version") != 1:
        raise TaskError("Unsupported NLP checkpoint schema")
    tokenizer, encoder = _encoder(state["encoder"])
    heads = TaskHeads(encoder.config.hidden_size).eval()
    heads.load_state_dict(state["heads"], strict=True)
    hidden, (offsets,) = _encode([text], tokenizer, encoder)
    with torch.no_grad():
        intent_logits, slot_logits = heads(hidden)
        intent_score, intent_id = intent_logits.softmax(1)[0].max(0)
        scores, tags = slot_logits.softmax(2)[0].max(1)
    if intent_score.item() < 0.80 or INTENTS[intent_id.item()] == "hold":
        raise TaskError("NLP intent requires clarification")

    spans = {}
    previous = None
    for (start, end), tag_id, score in zip(offsets, tags.tolist(), scores.tolist(), strict=True):
        tag = TAGS[tag_id]
        if start == end or tag == "O":
            previous = None
            continue
        prefix, name = tag.split("-", 1)
        if (
            score < 0.80
            or (prefix == "I" and previous != name)
            or (prefix == "B" and name in spans)
        ):
            raise TaskError("NLP slot requires clarification")
        spans[name] = (spans[name][0] if prefix == "I" else start, end)
        previous = name
    return {
        "intent": INTENTS[intent_id.item()],
        "slots": {name: text[start:end] for name, (start, end) in spans.items()},
    }


def resolve_text(text: str, checkpoint: Path, observation: Path) -> TaskRequest:
    prediction = predict_text(text, checkpoint)
    skill, slots = prediction["intent"], prediction["slots"]
    if skill == "serve_recipe":
        from .tabletop_model import frontend_references

        frontend = Path(__file__).resolve().parent.parent / "working/frontend"
        recipe = _normalized(slots.get("recipe", ""))
        ids = {
            row.class_id
            for row in frontend_references(frontend)
            if row.kind == "recipe" and recipe in {_normalized(row.name), row.class_id}
        }
        if len(ids) != 1:
            raise TaskError("NLP recipe does not match the fixed frontend catalog")
        raise TaskError(
            f"Recipe {ids.pop()} identified. Only bottle skills are wired to G1 planning"
        )
    if "recipe" in slots or _normalized(slots.get("destination", "原位")) not in {
        "原位",
        "原处",
        "桌面",
        "桌上",
        "原来的位置",
    }:
        raise TaskError("This bottle skill supports returning the bottle to its original position")
    if _normalized(slots.get("category", "")) not in {
        "水瓶",
        "塑料水瓶",
        "瓶子",
        "矿泉水瓶",
        "bottle",
    }:
        raise TaskError("NLP target category requires clarification")
    attr = _normalized(slots.get("attribute", ""))
    matches = [
        obj.object_id
        for obj in load_scene(observation).detections.values()
        if obj.category == "bottle" and attr in _normalized(obj.name + " " + obj.description)
    ]
    if len(matches) != 1:
        raise TaskError("NLP target must match exactly one bottle in the current scene")
    raw_count = _normalized(slots.get("repeat_count", ""))
    counts = {"一": 1, "二": 2, "两": 2, "三": 3, "1": 1, "2": 2, "3": 3}
    count = counts.get(raw_count.removesuffix("次").removesuffix("下"))
    if raw_count and count is None:
        raise TaskError("NLP repeat count requires clarification")
    return TaskRequest(skill, object_id=matches[0], repeat_count=count)
