"""Ten-second G1 show, adapted from GalbotSDK's Apache-2.0 head-sway example."""

import argparse
import json
import math
import signal
import time
import uuid
import wave
from array import array
from pathlib import Path

SHOW_SECONDS = 10
PCM_RATE = 16_000
PCM_CHUNK = 2560  # 80 ms of mono signed 16-bit PCM
PRESET_CLIP = Path("artifacts/g1-show/cctv-zu-hai-hao-yun-lai-10s.wav")
KEYFRAMES = ((0, 0.0), (2.5, 0.10), (7.5, -0.10), (10, 0.0))
SOURCE = (
    "https://github.com/GalaxyGeneralRobotics/GalbotSDK/blob/main/examples/"
    "g1/python/galbot_robot/installation_welcome_verify.py"
)
CHAINS = ("leg", "head", "left_arm", "right_arm")


def show_manifest():
    return {
        "duration_s": SHOW_SECONDS,
        "head_yaw_rad": KEYFRAMES,
        "choreography_source": SOURCE,
        "source_license": "Apache-2.0",
        "dispatch_allowed": False,
    }


def yaw_offset(seconds):
    for (start, initial), (end, target) in zip(KEYFRAMES, KEYFRAMES[1:]):
        if seconds <= end:
            phase = max(0.0, (seconds - start) / (end - start))
            return initial + (target - initial) * (1 - math.cos(math.pi * phase)) / 2
    return 0.0


def read_clip(path):
    with wave.open(str(path), "rb") as clip:
        if (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) != (1, 2, PCM_RATE):
            raise ValueError("Use mono 16 kHz signed 16-bit WAV")
        audio = clip.readframes(SHOW_SECONDS * PCM_RATE)
    if len(audio) != SHOW_SECONDS * PCM_RATE * 2:
        raise ValueError("The audio clip must contain at least ten seconds")
    return audio


def preview_volume(audio):
    samples = array("h")
    samples.frombytes(audio)
    # Leave the robot's global volume intact and cap the PCM signal for the first live test.
    return array("h", (sample // 4 for sample in samples)).tobytes()


def measured_joints(robot, names):
    rows = robot.get_joint_states([], names)
    now = time.time_ns()
    if [row.joint_name for row in rows] != names or any(
        not (0 <= now - row.timestamp_ns <= 250_000_000)
        or not math.isfinite(row.position)
        or not math.isfinite(row.velocity)
        or abs(row.velocity) > 0.03
        for row in rows
    ):
        raise RuntimeError("Fresh, stationary joint feedback is required")
    return [row.position for row in rows]


def localized_base(robot):
    base, stamp = robot.get_transform("map", "base_link", 0, 350)
    if len(base) != 7 or not all(math.isfinite(v) for v in base) or not (
        0 <= time.time_ns() - stamp <= 250_000_000
        and abs(sum(v * v for v in base[3:]) - 1) < 0.005
    ):
        raise RuntimeError("Fresh map-to-base localization is required for the collision scene")
    velocity = robot.get_base_velocity()
    for key in ("linear_velocity", "angular_velocity"):
        values = velocity.get(key, [])
        if len(values) != 3 or any(not math.isfinite(v) or abs(v) > 0.02 for v in values):
            raise RuntimeError("The base must be stationary")
    return list(base)


def check_scene(robot, motion, sdk, obstacles):
    if not obstacles or "ground" in obstacles:
        raise RuntimeError("Name the reviewed non-ground obstacles in the show area")
    loaded = set(motion.get_built_obstacles_list())
    if not set(obstacles) <= loaded:
        raise RuntimeError(f"Scene obstacles missing: {sorted(set(obstacles) - loaded)}")
    base = localized_base(robot)
    chains = [list(motion.get_chain_joint_names(chain)) for chain in CHAINS]
    if [len(chain) for chain in chains] != [5, 2, 7, 7] or "head_joint1" not in chains[1]:
        raise RuntimeError("The planner joint layout does not match this G1")
    names = [name for chain in chains for name in chain]
    start = measured_joints(robot, names)
    yaw = names.index("head_joint1")
    if abs(start[yaw]) > 0.55:
        raise RuntimeError("Head yaw is outside the starting envelope for this show")
    points = []
    for step in range(SHOW_SECONDS * 10 + 1):
        timestamp = step / 10
        pose = start.copy()
        pose[yaw] += yaw_offset(timestamp)
        state = sdk.RobotStates()
        state.whole_body_joint = pose
        state.base_state = base
        points.append((timestamp, state, pose[yaw]))
    for offset in range(0, len(points), 20):
        batch = [row[1] for row in points[offset:offset + 20]]
        status, hits = motion.check_collision(batch, True)
        if status != sdk.MotionStatus.SUCCESS or len(hits) != len(batch) or any(hits):
            raise RuntimeError(f"Show collision check rejected samples at index {offset}: {status}")
    return names, start, base, [(stamp, position) for stamp, _, position in points]


def stream_audio(robot, audio, stream_id, began):
    for offset in range(0, len(audio), PCM_CHUNK):
        due = began + offset / (PCM_RATE * 2)
        remaining = due - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        if time.monotonic() - due > 0.15:
            raise RuntimeError("Audio dispatch lost its timing window")
        if not robot.write_audio_stream_output(audio[offset:offset + PCM_CHUNK], stream_id):
            raise RuntimeError("G1 rejected an audio chunk")
    time.sleep(max(0, began + SHOW_SECONDS - time.monotonic()) + 0.12)


def play_audio(robot, audio):
    stream_id = "pet_meal_audio_" + uuid.uuid4().hex[:12]
    try:
        stream_audio(robot, audio, stream_id, time.monotonic())
        return {"audio_seconds_sent": SHOW_SECONDS, "motion_commands_sent": 0}
    finally:
        robot.stop_audio_stream_output(stream_id)


def play_show(robot, sdk, audio, names, start, base, points):
    trajectory = sdk.Trajectory()
    trajectory.joint_names = ["head_joint1"]
    trajectory.points = []
    commands = []
    for timestamp, position in points:
        point = sdk.TrajectoryPoint()
        point.time_from_start_second = max(0.001, timestamp)
        command = sdk.JointCommand()
        command.position = position
        point.joint_command_vec = [command]
        commands.append(point)
    trajectory.points = commands
    if math.dist(base, localized_base(robot)) > 0.01:
        raise RuntimeError("Base localization changed after collision checks")
    if max(abs(a - b) for a, b in zip(start, measured_joints(robot, names), strict=True)) > 0.01:
        raise RuntimeError("The start posture changed after collision checks")
    stream_id = "pet_meal_show_" + uuid.uuid4().hex[:12]
    stopped = True
    try:
        began = time.monotonic()
        result = robot.execute_joint_trajectory(trajectory, False)
        if result != sdk.ControlStatus.SUCCESS:
            stopped = False
            raise RuntimeError(f"G1 rejected show trajectory: {result}")
        stream_audio(robot, audio, stream_id, began)
        end = measured_joints(robot, names)
        if max(abs(a - b) for a, b in zip(start, end, strict=True)) > 0.02:
            raise RuntimeError("G1 did not return to the starting posture")
        stopped = False
        return {"audio_seconds_sent": SHOW_SECONDS, "endpoint_verified": True}
    finally:
        try:
            if stopped:
                robot.stop_trajectory_execution()
        finally:
            robot.stop_audio_stream_output(stream_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, help="10-second mono 16 kHz signed 16-bit WAV")
    parser.add_argument("--scene-obstacle", action="append", default=[])
    dispatch = parser.add_mutually_exclusive_group()
    dispatch.add_argument("--execute", action="store_true", help="Run with an on-site operator")
    dispatch.add_argument("--audio-only", action="store_true", help="Test only the speaker")
    args = parser.parse_args()
    if (args.execute or args.audio_only) and not args.wav:
        parser.error("Playback requires --wav")

    import galbot_sdk.g1 as sdk

    def interrupt(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    audio = preview_volume(read_clip(args.wav)) if args.wav else None
    robot = sdk.GalbotRobot()
    try:
        if not robot.init(set(), True):
            raise RuntimeError("G1 SDK initialization failed")
        if args.audio_only:
            print(json.dumps(play_audio(robot, audio)))
            return
        localized_base(robot)
        motion = sdk.GalbotMotion()
        if not motion.init():
            raise RuntimeError("G1 motion planner initialization failed")
        names, start, base, points = check_scene(robot, motion, sdk, args.scene_obstacle)
        result = {"collision_samples": len(points), "executed": False}
        if args.execute:
            result.update(play_show(robot, sdk, audio, names, start, base, points), executed=True)
        print(json.dumps(result))
    finally:
        robot.request_shutdown()
        robot.wait_for_shutdown()
        robot.destroy()


if __name__ == "__main__":
    main()
