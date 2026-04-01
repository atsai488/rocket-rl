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

from __future__ import annotations


class BNO055:
    """Read accel, gyro, and magnetometer from a BNO055 over I2C."""

    def __init__(self):
        """Open I2C bus on Jetson pins 3 (SDA) and 5 (SCL) and init the sensor."""
        self._sensor = None
        self._fallback = (0.0, 0.0, 0.0)

        try:
            # Lazy-import so environments without physical board support can still run.
            import board
            import busio
            import adafruit_bno055

            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_bno055.BNO055_I2C(i2c)
        except Exception as exc:
            print(f"[WARN] IMU unavailable, using zeroed readings: {exc}")

    def read_accel(self) -> tuple[float, float, float]:
        """Return linear acceleration (ax, ay, az) in m/s²."""
        if self._sensor is None:
            return self._fallback
        return self._sensor.acceleration

    def read_gyro(self) -> tuple[float, float, float]:
        """Return angular velocity (gx, gy, gz) in rad/s."""
        if self._sensor is None:
            return self._fallback
        return self._sensor.gyro

    def read_mag(self) -> tuple[float, float, float]:
        """Return magnetic field (mx, my, mz) in µT."""
        if self._sensor is None:
            return self._fallback
        return self._sensor.magnetic

    def read_all(self) -> dict:
        """Return all readings as a dict compatible with RocketState."""
        return {
            "accel": self.read_accel(),
            "gyro":  self.read_gyro(),
            "mag":   self.read_mag(),
        }
