import numpy as np
import time

class RocketState:
    def __init__(self):
        self.timestamp = None

        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)

        self.motor_angle = np.zeros(6)
        self.motor_velocity = np.zeros(6)

        self._last_motor_angle = np.zeros(6)

    def update_from_udp(self, state_dict):
        now = time.time()

        if self.timestamp is None:
            dt = None
        else:
            dt = now - self.timestamp

        self.timestamp = now

        # TODO these will be dependent on where I pack joint positions within the array when sending it back from the stm32/atmega
        # ----- Motors -----
        new_angles = np.array(state_dict["joints"], dtype=float)
        self.motor_angle = new_angles

        if dt and dt > 1e-6:
            self.motor_velocity = (new_angles - self._last_motor_angle) / dt

        self._last_motor_angle = new_angles.copy()

        # ----- IMU -----
        imu = state_dict["imu"]
        self.accel = np.array(imu[0:3])
        self.gyro  = np.array(imu[3:6])
        self.mag   = np.array(imu[6:9])