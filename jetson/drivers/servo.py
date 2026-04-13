#!/usr/bin/env python3
import time
import Jetson.GPIO as GPIO

PWM_FREQ: float = 50.0
MIN_US: int = 500
MAX_US: int = 2500
MIN_ANGLE: float = 0.0
MAX_ANGLE: float = 180.0
DEFAULT_SERVO_PIN: int = 33


class ServoController:
    def __init__(self, pin: int = DEFAULT_SERVO_PIN):
        self.pin = pin
        self._pwm = None
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT)

        self._pwm = GPIO.PWM(self.pin, PWM_FREQ)
        self._pwm.start(0.0)

    def angle_to_duty_cycle(self, angle: float) -> float:
        angle = max(MIN_ANGLE, min(MAX_ANGLE, float(angle)))
        pulse_us = MIN_US + (
            (angle - MIN_ANGLE)
            / (MAX_ANGLE - MIN_ANGLE)
        ) * (MAX_US - MIN_US)

        period_us = 1_000_000.0 / PWM_FREQ
        duty_cycle = (pulse_us / period_us) * 100.0
        return duty_cycle

    def set_angle(self, angle: float, settle_time: float = 0.01) -> None:
        """Move servo to an absolute angle in degrees."""
        duty = self.angle_to_duty_cycle(angle)
        self._pwm.ChangeDutyCycle(duty)
        time.sleep(settle_time)

    def hold(self, angle: float) -> None:
        """Set angle and keep pulses active."""
        duty = self.angle_to_duty_cycle(angle)
        self._pwm.ChangeDutyCycle(duty)

    def stop(self) -> None:
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(0.0)
            self._pwm.stop()
            self._pwm = None
        GPIO.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


# Example RL-friendly usage
if __name__ == "__main__":
    SERVO_PIN = DEFAULT_SERVO_PIN  # BOARD numbering

    with ServoController(pin=SERVO_PIN) as servo:
        servo.set_angle(90)
        time.sleep(1)

        servo.set_angle(30)
        time.sleep(1)

        servo.set_angle(150)
        time.sleep(1)