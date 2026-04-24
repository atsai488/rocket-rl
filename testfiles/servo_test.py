import math
import os
import time
from typing import Union
import Jetson.GPIO as GPIO

PWM_FREQ: float = 50.0
MIN_US: int = 500
MAX_US: int = 2500
MIN_ANGLE_RAD: float = 0.0
MAX_ANGLE_RAD: float = math.pi
DEFAULT_SERVO_PIN: int = 33


def resolve_pwm_channel(pin: int, gpio_mode: str) -> Union[int, str]:
    mode = gpio_mode.upper()
    if mode == "BOARD":
        return pin
    if mode == "TEGRA_SOC":
        return "GPIO_PE6" if pin == 33 else "LCD_BL_PW"
    raise ValueError("SERVO_GPIO_MODE must be BOARD or TEGRA_SOC")


class ServoController:
    def __init__(self, pin: int = DEFAULT_SERVO_PIN, gpio_mode: str = "BOARD"):
        self.pin = pin
        self.gpio_mode = gpio_mode.upper()
        self._pwm = None
        if self.gpio_mode == "BOARD":
            GPIO.setmode(GPIO.BOARD)
        elif self.gpio_mode == "TEGRA_SOC":
            GPIO.setmode(GPIO.TEGRA_SOC)
        else:
            raise ValueError("gpio_mode must be BOARD or TEGRA_SOC")

        self.channel = resolve_pwm_channel(pin, self.gpio_mode)
        GPIO.setup(self.channel, GPIO.OUT)
        self._pwm = GPIO.PWM(self.channel, PWM_FREQ)
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
        print(f"[DEBUG] angle={math.degrees(angle):.1f}deg  duty={duty:.2f}%  pulse={MIN_US + (angle/math.pi)*(MAX_US-MIN_US):.0f}us")
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


def main() -> None:
    pin = int(os.getenv("SERVO_PIN", str(DEFAULT_SERVO_PIN)))
    gpio_mode = os.getenv("SERVO_GPIO_MODE", "BOARD").upper()
    hold_s = float(os.getenv("SERVO_HOLD_S", "5.0"))
    start_deg = float(os.getenv("SERVO_START_DEG", "0.0"))
    target_deg = float(os.getenv("SERVO_TARGET_DEG", "15.0"))
    settle_s = float(os.getenv("SERVO_SETTLE_S", "1.0"))
    start_rad = math.radians(start_deg)
    target_rad = math.radians(target_deg)

    print(
        f"Servo test mode={gpio_mode} pin={pin}: {start_deg:.1f} deg -> {target_deg:.1f} deg, hold={hold_s:.1f}s"
    )
    with ServoController(pin=pin, gpio_mode=gpio_mode) as servo:
        servo.hold(start_rad)
        time.sleep(settle_s)
        servo.hold(target_rad)
        time.sleep(hold_s)

    print("Servo test complete")


if __name__ == "__main__":
    main()
        
        
