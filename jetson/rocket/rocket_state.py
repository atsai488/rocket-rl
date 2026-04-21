import numpy as np
import time
from threading import Lock

class RocketState:
    def __init__(self):

        self.timestamp = None

        # IMU
        self.accel = np.zeros(3)
        self.gyro = np.zeros(3)
        self.mag = np.zeros(3)
        self.quat = np.zeros(4)

        # Motors
        self.servo_motor_angle = np.zeros(2)
        self.stepper_motor_angle = np.zeros(4)
        self.motor_velocity = np.zeros(4)

        self._last_motor_angle = np.zeros(6)

    def update_servo_position(self, state_dict):
        self.servo_motor_angle = np.array(state_dict["joints"], dtype=float)
    
    def update_from_single_encoder(self, addr, data):
        self.stepper_motor_angle[addr-1] = data
        self._last_motor_angle = self.stepper_motor_angle.copy()
    
    def update_from_encoders(self, state_dict):
        now = time.perf_counter()

        if self.timestamp is None:
            dt = None
        else:
            dt = now - self.timestamp

        self.timestamp = now
        self.stepper_motor_angle = np.array(state_dict["joints"], dtype=float)
        
        if dt and dt > 1e-6:
            self.motor_velocity = (self.stepper_motor_angle - self._last_motor_angle) / dt
        self._last_motor_angle = self.stepper_motor_angle.copy()

    def update_from_imu(self, imu_data: dict) -> None:
        self.accel = np.array(imu_data["accel"])
        self.gyro = np.array(imu_data["gyro"])
        self.mag = np.array(imu_data["mag"])
        self.quat = np.array(imu_data["quaternion"])

    def to_observation(self):
        obs = np.concatenate(
            [self.stepper_motor_angle, self.motor_velocity, self.gyro, self.accel, self.quat]
        )

        return obs
