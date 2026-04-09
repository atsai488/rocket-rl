import numpy as np
import time
from threading import Lock

class RocketState:
    def __init__(self):

        self.lock = Lock()
        self.timestamp = None

        # IMU
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.quat = np.zeros(4)

        # Motors
        self.motor_angle = np.zeros(6)
        self.motor_velocity = np.zeros(6)

        self._last_motor_angle = np.zeros(6)

    def update_from_atmega(self, state_dict):
        now = time.perf_counter()

        with self.lock:
            if self.timestamp is None:
                dt = None
            else:
                dt = now - self.timestamp

            self.timestamp = now

            # TODO these will need to be double checked when we pack the data from the stm32/atmega
            # ---- Motors ----
            new_angles = np.array(state_dict["joints"], dtype=float)
            self.motor_angle = new_angles

            if dt and dt > 1e-6:
                self.motor_velocity = (new_angles - self._last_motor_angle) / dt

            self._last_motor_angle = new_angles.copy()

    def update_from_imu(self, imu_data: dict) -> None:
        with self.lock:
            self.accel = np.array(imu_data["accel"])
            self.gyro = np.array(imu_data["gyro"])
            self.mag = np.array(imu_data["mag"])
            self.quat = np.array(imu_data["quaternion"])

    def to_observation(self):
        with self.lock:
            print(self.motor_angle.shape, self.motor_velocity.shape, self.gyro.shape, self.accel.shape, self.quat.shape)
            obs = np.concatenate(
                [self.motor_angle, self.motor_velocity, self.gyro, self.accel, self.quat]
            )

        return obs
