import argparse
import sys
from pathlib import Path

from rocket.rocket import Rocket
from drivers.atmega_i2c import AtmegaI2C
from drivers.imu import BNO055
from rocket_onnx.onnx_command_generator import (
    RocketOnnxPositionController,
    RocketOnnxContext,
)
from utils.event_divider import EventDivider

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("policy_file_path", type=Path)
    parser.add_argument("--verbose", action="store_true")
    options = parser.parse_args()

    class Config:
        action_scale = 1.0
        num_joints = 6
        default_joints = [0.0] * 6
        verbose = options.verbose
        servo_left_pin = 7
        servo_right_pin = 8

    config = Config()
    rocket = Rocket(config)
    atmega = AtmegaI2C()
    imu = BNO055()
    context = RocketOnnxContext()
    # config = orbit.orbit_configuration.load_configuration(conf_file)
    print(config)

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
        rocket.start_state_loop(imu, context=context)
        
        input("Press ENTER to start command stream...")

        print("[INFO] Starting command stream...")
        rocket.start_command_stream(controller, timing_policy, atmega)

        input("Press ENTER to stop...")

    except KeyboardInterrupt:
        print("Interrupted")

    finally:
        print("[INFO] Shutting down...")
        atmega.close()

        print("[INFO] All stopped.")


if __name__ == "__main__":
    if not main():
        sys.exit(1)
