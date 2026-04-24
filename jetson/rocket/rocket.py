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
ACTION_DELAY_S = 0.05

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
            start = time.time()
            data = imu.read_all()
            context.latest_state.update_from_imu(data)
            end = time.time()
            print(f"[IMU]       t={end:.6f}  dt={end - start:.6f}s")
            enc_start = time.time()
            data = self.stepper_driver.read_all_joints()
            context.latest_state.update_from_encoders(data)
            enc_end = time.time()
            print(f"[ENCODER] t={enc_end:.6f}  dt={enc_end - enc_start:.6f}s")
            context.event.set()
        
    
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
        loop_count = 0
        totals = {"policy": 0.0, "send": 0.0, "sleep": 0.0, "total": 0.0}

        while not self._command_stream_stopping:
            if timing_policy():
                start_time = time.time()
                print(f"[LOOP START]   t={start_time:.6f}")

                policy_start = time.time()
                print(f"[POLICY START] t={policy_start:.6f}")
                cmd = command_policy()
                policy_end = time.time()
                print(f"[POLICY END]   t={policy_end:.6f}  dt={policy_end - policy_start:.6f}s")

                joint_angles = cmd.joint_angles
                if len(joint_angles) < 6:
                    self.logger.warning("command policy returned too few joint angles")
                    continue

                # added a safety clip to joint limits
                scaled_joint_angles = [
                    max(mid - scale, min(mid + angle * scale, mid + scale))
                    for mid, angle, scale in zip(JOINT_POS_MID, joint_angles, JOINT_SCALE)
                ]
                
                send_start = time.time()
                print(f"[SEND START]   t={send_start:.6f}")
                self.stepper_driver.send_joint_position(
                   {addr: scaled_joint_angles[addr+1] for addr in range(1, 5)}, 20)
                send_end = time.time()
                print(f"[SEND END]     t={send_end:.6f}  dt={send_end - send_start:.6f}s")

                # self.servo_driver_right.hold(scaled_joint_angles[0])
                # self.servo_driver_left.hold(scaled_joint_angles[1])
                self._started_streaming = True
                print("Sending command:", scaled_joint_angles, "\n\n")

                sleep_start = time.time()
                print(f"[SLEEP START]  t={sleep_start:.6f}")
                time.sleep(ACTION_DELAY_S)
                sleep_end = time.time()
                print(f"[SLEEP END]    t={sleep_end:.6f}  dt={sleep_end - sleep_start:.6f}s")

                end_time = time.time()
                print(f"[LOOP END]     t={end_time:.6f}  dt={end_time - start_time:.6f}s  (total)")
                print("time: ", end_time - start_time)

                loop_count += 1
                totals["policy"] += policy_end - policy_start
                totals["send"]   += send_end - send_start
                totals["sleep"]  += sleep_end - sleep_start
                totals["total"]  += end_time - start_time
                if loop_count % 5 == 0:
                    print(f"\n--- AVERAGES over {loop_count} loops ---")
                    for k, v in totals.items():
                        print(f"  {k:<8}: {v / loop_count:.6f}s")
                    print("---\n")
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

