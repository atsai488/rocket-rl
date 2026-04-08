"""
BNO055 I2C driver for Jetson.

Hardware wiring (Jetson 40-pin header):
  Pin 3  (SDA, I2C-1) → IMU SDA
  Pin 5  (SCL, I2C-1) → IMU SCL
  Pin 1  (3.3V)       → IMU VCC
  Pin 6  (GND)        → IMU GND

Requires:
  pip3 install adafruit-blinka adafruit-circuitpython-bno055

Notes:
- `acceleration` includes gravity.
- `linear_acceleration` removes gravity.
- `gyro` is in rad/s.
- `magnetic` is in microtesla.
- `euler` returns (heading, roll, pitch).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


Vector3 = Tuple[float, float, float]


class BNO055:
    """Read data from a BNO055 over I2C on Jetson."""

    def __init__(self) -> None:
        self._sensor: Optional[Any] = None
        self._fallback: Vector3 = (0.0, 0.0, 0.0)

        try:
            import board
            import busio
            import adafruit_bno055

            i2c = busio.I2C(board.SCL, board.SDA)
            self._sensor = adafruit_bno055.BNO055_I2C(i2c)
        except Exception as exc:
            print(f"[WARN] IMU unavailable, using zeroed readings: {exc}")

    def is_available(self) -> bool:
        """Return True if the sensor initialized successfully."""
        return self._sensor is not None

    def read_accel(self) -> Vector3:
        """Return total acceleration (ax, ay, az) in m/s^2."""
        if self._sensor is None:
            return self._fallback
        value = self._sensor.acceleration
        return value if value is not None else self._fallback

    def read_linear_accel(self) -> Vector3:
        """Return linear acceleration (gravity removed) in m/s^2."""
        if self._sensor is None:
            return self._fallback
        value = self._sensor.linear_acceleration
        return value if value is not None else self._fallback

    def read_gyro(self) -> Vector3:
        """Return angular velocity (gx, gy, gz) in rad/s."""
        if self._sensor is None:
            return self._fallback
        value = self._sensor.gyro
        return value if value is not None else self._fallback

    def read_mag(self) -> Vector3:
        """Return magnetic field (mx, my, mz) in microtesla."""
        if self._sensor is None:
            return self._fallback
        value = self._sensor.magnetic
        return value if value is not None else self._fallback

    def read_euler(self) -> Vector3:
        """Return Euler angles as (heading, roll, pitch) in degrees."""
        if self._sensor is None:
            return self._fallback
        value = self._sensor.euler
        return value if value is not None else self._fallback

    def read_quaternion(self) -> Tuple[float, float, float, float]:
        """Return quaternion as (w, x, y, z)."""
        if self._sensor is None:
            return (1.0, 0.0, 0.0, 0.0)
        value = self._sensor.quaternion
        if value is None:
            return (1.0, 0.0, 0.0, 0.0)
        return value

    def calibration_status(self) -> Dict[str, int]:
        """
        Return calibration status for sys, gyro, accel, and mag.
        Each value is 0-3.
        """
        if self._sensor is None:
            return {"sys": 0, "gyro": 0, "accel": 0, "mag": 0}

        try:
            sys_c, gyro_c, accel_c, mag_c = self._sensor.calibration_status
            return {
                "sys": int(sys_c),
                "gyro": int(gyro_c),
                "accel": int(accel_c),
                "mag": int(mag_c),
            }
        except Exception:
            return {"sys": 0, "gyro": 0, "accel": 0, "mag": 0}

    def read_all(self) -> Dict[str, Any]:
        """Return all readings in one dict."""
        return {
            "accel": self.read_accel(),
            "linear_accel": self.read_linear_accel(),
            "gyro": self.read_gyro(),
            "mag": self.read_mag(),
            "euler": self.read_euler(),
            "quaternion": self.read_quaternion(),
            "calibration": self.calibration_status(),
        }