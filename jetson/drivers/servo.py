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

"""

JETSON_NANO_PIN_DEFS = [
    (216,  '', "tegra-gpio", 7, 4, 'GPIO9', 'AUD_MCLK', None, None),
    (50,  '', "tegra-gpio", 11, 17, 'UART1_RTS', 'UART2_RTS', None, None),
    (79, '',  "tegra-gpio", 12, 18, 'I2S0_SCLK', 'DAP4_SCLK', None, None),
    (14,  '', "tegra-gpio", 13, 27, 'SPI1_SCK', 'SPI2_SCK', None, None),
    (194,  '', "tegra-gpio", 15, 22, 'GPIO12', 'LCD_TE', None, None),
    (232,  '', "tegra-gpio", 16, 23, 'SPI1_CS1', 'SPI2_CS1', None, None),
    (15,  '', "tegra-gpio", 18, 24, 'SPI1_CS0', 'SPI2_CS0', None, None),
    (16,  '', "tegra-gpio", 19, 10, 'SPI0_MOSI', 'SPI1_MOSI', None, None),
    (17,  '', "tegra-gpio", 21, 9, 'SPI0_MISO', 'SPI1_MISO', None, None),
    (13,  '', "tegra-gpio", 22, 25, 'SPI1_MISO', 'SPI2_MISO', None, None),
    (18,  '', "tegra-gpio", 23, 11, 'SPI0_SCK', 'SPI1_SCK', None, None),
    (19,  '', "tegra-gpio", 24, 8, 'SPI0_CS0', 'SPI1_CS0', None, None),
    (20,  '', "tegra-gpio", 26, 7, 'SPI0_CS1', 'SPI1_CS1', None, None),
    (149,  '', "tegra-gpio", 29, 5, 'GPIO01', 'CAM_AF_EN', None, None),
    (200,  '', "tegra-gpio", 31, 6, 'GPIO11', 'GPIO_PZ0', None, None),
    # Older versions of L4T have a DT bug which instantiates a bogus device
    # which prevents this library from using this PWM channel.
    (168,  '', "tegra-gpio", 32, 12, 'GPIO07', 'LCD_BL_PW', '7000a000.pwm', 0),
    (38,  '', "tegra-gpio", 33, 13, 'GPIO13', 'GPIO_PE6', '7000a000.pwm', 2),
    (76,  '', "tegra-gpio", 35, 19, 'I2S0_FS', 'DAP4_FS', None, None),
    (51,  '', "tegra-gpio", 36, 16, 'UART1_CTS', 'UART2_CTS', None, None),
    (12,  '', "tegra-gpio", 37, 26, 'SPI1_MOSI', 'SPI2_MOSI', None, None),
    (77,  '', "tegra-gpio", 38, 20, 'I2S0_DIN', 'DAP4_DIN', None, None),
    (78,  '', "tegra-gpio", 40, 21, 'I2S0_DOUT', 'DAP4_DOUT', None, None)
"""