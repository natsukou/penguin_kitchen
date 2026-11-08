import math
from types import SimpleNamespace

import pytest

from pet_meal import g1_show


def test_show_returns_head_to_start_and_stays_in_small_yaw_envelope():
    offsets = [g1_show.yaw_offset(t / 100) for t in range(1001)]
    assert offsets[0] == offsets[-1] == 0
    assert min(offsets) == -0.10 and max(offsets) == 0.10
    assert max(abs(a - b) for a, b in zip(offsets, offsets[1:])) * 100 < 0.07
    assert g1_show.show_manifest()["dispatch_allowed"] is False


def test_show_rejects_missing_environment_before_requesting_motion():
    motion = SimpleNamespace(get_built_obstacles_list=lambda: ["ground"])
    with pytest.raises(RuntimeError, match="Scene obstacles missing"):
        g1_show.check_scene(None, motion, None, ["verified_show_boundary"])


@pytest.mark.parametrize("reject", [False, True])
def test_audio_has_ten_seconds_of_samples_and_stops_its_stream(monkeypatch, reject):
    now = 0.0
    sent = []
    stopped = []

    def sleep(delay):
        nonlocal now
        now += delay

    def write(chunk, stream_id):
        sent.append((len(chunk), stream_id, now))
        return not reject

    monkeypatch.setattr(g1_show, "time", SimpleNamespace(monotonic=lambda: now, sleep=sleep))
    robot = SimpleNamespace(
        write_audio_stream_output=write,
        stop_audio_stream_output=stopped.append,
    )
    pcm = bytes(320_000)
    if reject:
        with pytest.raises(RuntimeError, match="rejected"):
            g1_show.play_audio(robot, pcm)
        assert len(sent) == 1
    else:
        result = g1_show.play_audio(robot, pcm)
        assert result == {"audio_seconds_sent": 10, "motion_commands_sent": 0}
        assert sum(row[0] for row in sent) == len(pcm) and len(sent) == 125
        assert math.isclose(sent[-1][2], 9.92)
    assert stopped == [sent[0][1]]


@pytest.mark.parametrize("fail_audio", [False, True])
def test_show_targets_head_only_and_stops_motion_if_audio_fails(monkeypatch, fail_audio):
    pose = [0, 0, 0, 0, 0, 0, 1]
    monkeypatch.setattr(g1_show, "localized_base", lambda _: pose)
    monkeypatch.setattr(g1_show, "measured_joints", lambda *_: [0.0])
    commands = []
    stopped = []

    def execute(trajectory, blocking):
        assert not blocking
        commands.append(trajectory)
        return "accepted"

    def stream(*_):
        if fail_audio:
            raise RuntimeError("speaker failed")

    monkeypatch.setattr(g1_show, "stream_audio", stream)
    robot = SimpleNamespace(
        execute_joint_trajectory=execute,
        stop_trajectory_execution=lambda: stopped.append("motion"),
        stop_audio_stream_output=lambda _: stopped.append("audio"),
    )
    sdk = SimpleNamespace(
        Trajectory=SimpleNamespace,
        TrajectoryPoint=SimpleNamespace,
        JointCommand=SimpleNamespace,
        ControlStatus=SimpleNamespace(SUCCESS="accepted"),
    )
    args = (robot, sdk, b"", ["head_joint1"], [0.0], pose, [(0, 0.0), (10, 0.0)])
    if fail_audio:
        with pytest.raises(RuntimeError, match="speaker failed"):
            g1_show.play_show(*args)
    else:
        assert g1_show.play_show(*args)["endpoint_verified"]
    assert len(commands) == 1 and commands[0].joint_names == ["head_joint1"]
    assert commands[0].points[-1].time_from_start_second == 10
    assert stopped == (["motion", "audio"] if fail_audio else ["audio"])
