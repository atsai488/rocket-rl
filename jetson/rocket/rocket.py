from threading import Thread
from typing import Callable
from rocket_onnx.onnx_command_generator import JointCommand
from rocket.rocket_state import RocketState
import logging

class Rocket:
    def __init__(self, config) -> None:
        self._started_streaming = False
        self._command_stream_stopping = False
        self._state_stream_stopping = False

        self._command_thread = None
        self._state_thread = None
        self.logger = logging.getLogger("Rocket")
        self.config = config
        if config.verbose:
            logging.basicConfig(level=logging.DEBUG)

    def __del__(self):
        """clean up active streams and threads if spot goes out of scope or is deleted"""
        self.stop_command_stream()


    def power_on(self):
        """Turn on power to robot's motors."""
        # TODO send message to turn on the enable signal to all the motors
        pass


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
            self._i2c_command_sender(command_policy, timing_policy, atmega)
        except Exception as e:
            self.logger.error(f"Error in command stream: {e}")
        finally:
            self.logger.info("Command stream stopped")
    
    def _i2c_command_sender(self, command_policy, timing_policy, atmega):
        """Send commands over i2c to the robot."""

        while not self._command_stream_stopping:
            if timing_policy():
                cmd = command_policy()
                atmega.send_command(cmd.joint_angles)
                
                self._started_streaming = True
                print("Sending command:", cmd.joint_angles)
            else:
                self.logger.warning("timing policy timeout")
                continue

    def stop_command_stream(self):
        """Stop sending joint commands to the robot."""
        if self._command_thread is not None:
            self._command_stream_stopping = True
            self._command_thread.join()


