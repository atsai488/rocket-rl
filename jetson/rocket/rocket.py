from threading import Thread
from typing import Callable
from rocket_onnx.onnx_command_generator import JointCommand
from rocket.rocket_state import RocketState
import logging
import time
from drivers.rs485 import Rs485Driver
from drivers.servo import ServoController

JOINT_POS_MID = [-0.0873, -0.0873, -0.0872, -0.0872, -0.0436, -0.0436]
JOINT_SCALE = [0.1745, 0.1745, 0.2618, 0.2618, 0.3054, 0.3054]

class Rocket:
    def __init__(self, config) -> None:
        
        self._started_streaming = False
        self._command_stream_stopping = False
        self._state_stream_stopping = False
        self._command_thread = None
        self._state_thread = None
        
        self.stepper_driver = Rs485Driver()
        self.servo_driver_left = ServoController(pin=33)
        self.servo_driver_right = ServoController(pin=32)
        
        self.logger = logging.getLogger("Rocket")
        self.config = config
        if config.verbose:
            logging.basicConfig(level=logging.DEBUG)

    def __del__(self):
        """clean up active streams and threads if spot goes out of scope or is deleted"""
        self.stop_command_stream()
        self.stop_state_stream();


    def power_on(self):
        """Turn on power to robot's motors."""
        # TODO send message to turn on the enable signal to all the motors
        pass

    def start_state_loop(self, imu, context):
        self._state_thread = Thread(
            target=self._run_state_loop, args=(imu, context), daemon=True
        )
        self._state_thread.start()
    
    
    
    def _run_state_loop(self, imu, context):
        while not self._state_stream_stopping:
            data = imu.read_all()
            context.latest_state.update_from_imu(data)
            data = self.stepper_driver.read_all_joints()
            context.latest_state.update_from_encoders(data)
            context.event.set()
            # print(context.latest_state.to_observation())
            time.sleep(0.005)
        
    
    def start_command_stream(self, command_policy, timing_policy, atmega):
        """Create command stream to send joint level commands to the robot.

        arguments:
        command_policy -- Callable that creates one joint command
        timing_policy -- Callable that blocks until the next time a command should be sent
        """
        self._command_thread = Thread(
            target=self._run_command_stream, args=(command_policy, timing_policy, atmega), daemon=True
        )
        self._command_thread.start()
    
    def _run_command_stream(
        self, command_policy: Callable[[None], JointCommand], timing_policy: Callable[[None], None], atmega
    ):
        """private function to be run in command stream thread.

        arguments
        command_policy -- callback supplied to start_command_stream to create commands
        timing_policy -- callback supplied to start_command_stream to control timing
        """
        try:
            self.logger.info("Starting command stream")
            self._command_snder(command_policy, timing_policy, atmega)
        except Exception as e:
            self.logger.error(f"Error in command stream: {e}")
        finally:
            self.logger.info("Command stream stopped")
    
    
    def _command_snder(self, command_policy, timing_policy, atmega):
        """Send commands over i2c to the robot."""

        while not self._command_stream_stopping:
            if timing_policy():
                cmd = command_policy()
                joint_angles = cmd.joint_angles
                if len(joint_angles) < 6:
                    self.logger.warning("command policy returned too few joint angles")
                    continue

                scaled_joint_angles = [
                    mid + angle * scale
                    for mid, angle, scale in zip(JOINT_POS_MID, joint_angles, JOINT_SCALE)
                ]

                # self.stepper_driver.send_joint_position(
                #    {addr: scaled_joint_angles[addr + 1] for addr in range(1, 5)}
                # )
                # self.servo_driver_right.hold(scaled_joint_angles[0])
                # self.servo_driver_left.hold(scaled_joint_angles[1])
                # self._started_streaming = True
                print("Sending command:", scaled_joint_angles)
            else:
                self.logger.warning("timing policy timeout")
                continue

    def stop_command_stream(self):
        """Stop sending joint commands to the robot."""
        if self._command_thread is not None:
            self._command_stream_stopping = True
            self._command_thread.join()
    
    def stop_state_stream(self):
        if self._state_thread is not None:
            self._state_stream_stopping = True
            self._state_thread.join()

