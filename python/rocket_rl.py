import argparse
import sys
import time
from pathlib import Path
from threading import Event


from rocket.rocket import Rocket
from rocket.rocket_state import RocketState
from rocket_onnx.onnx_command_generator import (
    RocketOnnxPositionController,
    RocketOnnxContext,
    StateHandler,
)
from utils.event_divider import EventDivider

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("--robot-ip", type=str, required=True)
    parser.add_argument("--command-port", type=int, default=15001)
    parser.add_argument("--state-port", type=int, default=15002)
    parser.add_argument("--state-msg-size", type=int, default=4 * (6 + 9))
    parser.add_argument("--verbose", action="store_true")
    options = parser.parse_args()
    class Config:
        robot_ip = options.robot_ip
        command_port = options.command_port
        state_port = options.state_port
        STATE_MSG_SIZE = options.state_msg_size

        action_scale = 1.0
        num_joints = 6
        default_joints = [0.0] * 6
        verbose = options.verbose

    config = Config()
    rocket = Rocket(config)
    rocket_state = RocketState()
    context = RocketOnnxContext()
    # config = orbit.orbit_configuration.load_configuration(conf_file)
    print(config)

    state_handler = StateHandler(context)
    print(options.verbose)

    # 333 Hz state update / 6 => ~56 Hz control updates
    timing_policy = EventDivider(context.event, 6)
    

    controller = RocketOnnxPositionController(
        context=context,
        config=config,
        model_path=str(options.policy_file_path),
        verbose=options.verbose,
    )


    try:
        print("[INFO] Starting state stream...")
        rocket.start_state_stream(state_handler)

        input("Press ENTER to start command stream...")

        print("[INFO] Starting command stream...")
        rocket.start_command_stream(controller, timing_policy)

        input("Press ENTER to stop...")

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        print("[INFO] Stopping command stream...")
        rocket.stop_command_stream()

        print("[INFO] Stopping state stream...")
        rocket.stop_state_stream()

        print("[INFO] All stopped.")


if __name__ == "__main__":
    if not main():
        sys.exit(1)