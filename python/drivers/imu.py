"""
BNO055 I2C driver for Jetson.

Hardware wiring (Jetson 40-pin header):
  Pin 3  (SDA, I2C-1) → IMU SDA
  Pin 5  (SCL, I2C-1) → IMU SCL
  Pin 1  (3.3V)       → IMU VCC
  Pin 6  (GND)        → IMU GND

The Adafruit BNO055 library handles all register communication internally.
Requires adafruit-blinka and adafruit-circuitpython-bno055.

read_all() returns:
  {
    "accel": (x, y, z)  m/s²
    "gyro":  (x, y, z)  rad/s
    "mag":   (x, y, z)  µT
  }
"""

import board
import busio
import adafruit_bno055


class BNO055:
    """Read accel, gyro, and magnetometer from a BNO055 over I2C."""

    def __init__(self):
        """Open I2C bus on Jetson pins 3 (SDA) and 5 (SCL) and init the sensor."""
        i2c = busio.I2C(board.SCL, board.SDA)
        self._sensor = adafruit_bno055.BNO055_I2C(i2c)

    def read_accel(self) -> tuple[float, float, float]:
        """Return linear acceleration (ax, ay, az) in m/s²."""
        return self._sensor.acceleration

    def read_gyro(self) -> tuple[float, float, float]:
        """Return angular velocity (gx, gy, gz) in rad/s."""
        return self._sensor.gyro

    def read_mag(self) -> tuple[float, float, float]:
        """Return magnetic field (mx, my, mz) in µT."""
        return self._sensor.magnetic

    def read_all(self) -> dict:
        """Return all readings as a dict compatible with RocketState."""
        return {
            "accel": self.read_accel(),
            "gyro":  self.read_gyro(),
            "mag":   self.read_mag(),
        }
