import math
import time
import Jetson.GPIO as GPIO

PWM_FREQ: float = 50.0
MIN_US: int = 500
MAX_US: int = 2500
MIN_ANGLE_RAD: float = 0.0
MAX_ANGLE_RAD: float = math.pi
DEFAULT_SERVO_PIN: int = 33


class ServoController:
    def __init__(self, pin: int = DEFAULT_SERVO_PIN):
        self.pin = pin
        self._pwm = None
        soc_name = "LCD_BL_PW"
        if (pin == 33):
            soc_name = "GPIO_PE6"

        GPIO.setup(soc_name, GPIO.OUT)
        self._pwm = GPIO.PWM(soc_name, PWM_FREQ)
        self._pwm.start(0.0)

    def angle_to_duty_cycle(self, angle: float) -> float:
        """Convert servo angle in radians to duty cycle."""
        angle = max(MIN_ANGLE_RAD, min(MAX_ANGLE_RAD, float(angle)))
        pulse_us = MIN_US + (
            (angle - MIN_ANGLE_RAD)
            / (MAX_ANGLE_RAD - MIN_ANGLE_RAD)
        ) * (MAX_US - MIN_US)

        period_us = 1_000_000.0 / PWM_FREQ
        duty_cycle = (pulse_us / period_us) * 100.0
        return duty_cycle

    def hold(self, angle: float) -> None:
        """Set angle in radians and keep pulses active."""
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
        
        
