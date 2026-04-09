import argparse
import sys
import time
from pathlib import Path
from threading import Event
from threading import Thread

from rocket.rocket import Rocket
from drivers.atmega_i2c import AtmegaI2C
from drivers.imu import BNO055
from rocket_onnx.onnx_command_generator import (
    RocketOnnxPositionController,
    RocketOnnxContext,
)
from utils.event_divider import EventDivider


def _imu_loop(imu: BNO055, context: RocketOnnxContext) -> None:
    while True:
        data = imu.read_all()
        context.latest_state.update_from_imu(data)
        data = {"joints": [0.0, 0.0, 0.0, 0.0]} # TODO: Read serial from Encoders
        context.latest_state.update_from_encoders(data)
        context.event.set()
        time.sleep(0.00833)


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
        imu_thread = Thread(target=lambda: _imu_loop(imu, context), daemon=True)
        imu_thread.start()

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
