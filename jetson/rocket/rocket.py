from threading import Thread
from typing import Callable
from rocket_onnx.onnx_command_generator import JointCommand
from rocket.rocket_state import RocketState
import logging
import time
from drivers.rs485 import Rs485Driver
from drivers.servo import ServoController
from rocket.constants import JOINT_MID, JOINT_HALF_RANGE, JOINT_FULL_RANGE

ACTION_DELAY_S = 0.08   # 0.08 -> 10 hz, 0.02 --> 25hz

class Rocket:
    def __init__(self, config, policy_type: str = "delta") -> None:
        if policy_type not in ("delta", "position"):
            raise ValueError(f"Unknown policy_type '{policy_type}'. Choose 'delta' or 'position'.")
        self.policy_type = policy_type

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
        print(f"[Rocket] policy_type={policy_type}")
        print(f"[Rocket] JOINT_MID        = {[round(v, 4) for v in JOINT_MID]}")
        print(f"[Rocket] JOINT_HALF_RANGE = {[round(v, 4) for v in JOINT_HALF_RANGE]}")
        print(f"[Rocket] JOINT_FULL_RANGE = {[round(v, 4) for v in JOINT_FULL_RANGE]}")

    def __del__(self):
        self.stop_command_stream()
        self.stop_state_stream()

    def power_on(self):
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
        try:
            self.logger.info("Starting command stream")
            self._command_snder(command_policy, timing_policy, atmega)
        except Exception as e:
            self.logger.error(f"Error in command stream: {e}")
        finally:
            self.logger.info("Command stream stopped")

    def _command_snder(self, command_policy, timing_policy, atmega):
        loop_count = 0
        totals = {"policy": 0.0, "send": 0.0, "sleep": 0.0, "total": 0.0}

        n_motors = 6
        prev_pos = None
        prev_vel = None
        prev_acc = None
        sum_abs_vel = [0.0] * n_motors
        sum_abs_acc = [0.0] * n_motors
        sum_abs_jrk = [0.0] * n_motors

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

                if self.policy_type == "delta":
                    # action × full_range = displacement in radians; rs485 adds to current + clamps
                    raw_angles = [a * r for a, r in zip(joint_angles, JOINT_FULL_RANGE)]
                else:
                    # mid + action × half_range = absolute target position
                    raw_angles = [mid + a * hr for mid, a, hr in zip(JOINT_MID, joint_angles, JOINT_HALF_RANGE)]

                if prev_pos is not None:
                    delta_pos = [raw_angles[i] - prev_pos[i] for i in range(n_motors)]
                    for i in range(n_motors):
                        sum_abs_vel[i] += abs(delta_pos[i])
                    if prev_vel is not None:
                        delta_vel = [delta_pos[i] - prev_vel[i] for i in range(n_motors)]
                        for i in range(n_motors):
                            sum_abs_acc[i] += abs(delta_vel[i])
                        if prev_acc is not None:
                            delta_acc = [delta_vel[i] - prev_acc[i] for i in range(n_motors)]
                            for i in range(n_motors):
                                sum_abs_jrk[i] += abs(delta_acc[i])
                        prev_acc = delta_vel
                    else:
                        prev_acc = None
                        delta_vel = None
                    prev_vel = delta_pos
                else:
                    prev_vel = None
                    prev_acc = None
                prev_pos = raw_angles

                send_start = time.time()
                print(f"[SEND START]   t={send_start:.6f}")
                self.stepper_driver.send_joint_position_fast(
                    {addr: raw_angles[addr + 1] for addr in range(1, 5)},
                    hip_pulses=10, knee_pulses=20,
                    delta=(self.policy_type == "delta"))
                send_end = time.time()
                print(f"[SEND END]     t={send_end:.6f}  dt={send_end - send_start:.6f}s")

                self._started_streaming = True

                sleep_start = time.time()
                print(f"[SLEEP START]  t={sleep_start:.6f}")
                time.sleep(ACTION_DELAY_S)
                sleep_end = time.time()
                print(f"[SLEEP END]    t={sleep_end:.6f}  dt={sleep_end - sleep_start:.6f}s")

                end_time = time.time()
                print(f"[LOOP END]     t={end_time:.6f}  dt={end_time - start_time:.6f}s  (total)")

                loop_count += 1
                totals["policy"] += policy_end - policy_start
                totals["send"]   += send_end - send_start
                totals["sleep"]  += sleep_end - sleep_start
                totals["total"]  += end_time - start_time
                if loop_count % 5 == 0:
                    avg_total = totals["total"] / loop_count
                    print(f"\n--- AVERAGES over {loop_count} loops ---")
                    for k, v in totals.items():
                        print(f"  {k:<8}: {v / loop_count:.6f}s")
                    print(f"  hz      : {1 / avg_total:.2f} Hz")
                    vel_denom = max(loop_count - 1, 1)
                    acc_denom = max(loop_count - 2, 1)
                    jrk_denom = max(loop_count - 3, 1)
                    print(f"  {'motor':<6}  {'avg|vel|':>10}  {'avg|acc|':>10}  {'avg|jerk|':>10}")
                    for i in range(n_motors):
                        print(f"  m{i:<5}  {sum_abs_vel[i]/vel_denom:>10.6f}  {sum_abs_acc[i]/acc_denom:>10.6f}  {sum_abs_jrk[i]/jrk_denom:>10.6f}")
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
