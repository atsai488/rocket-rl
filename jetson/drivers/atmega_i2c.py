import struct
import time
from threading import Thread, Lock
from typing import Callable

try:
    import smbus2  # type: ignore[import-not-found]
except ImportError:
    print("Import not found")
    smbus2 = None

ATMEGA_ADDR = 0x08  # i2c address of the atmega
NUM_JOINTS = 6


class AtmegaI2C:
    def __init__(
        self, bus: int = 1, addr: int = ATMEGA_ADDR, poll_interval: float = 0.01
    ):
        self._bus = None
        self._addr = addr
        self._poll_interval = poll_interval
        self._pending_cmd = None
        self._lock = Lock()
        self._thread = None
        self._stopping = False
        self._fallback_state = tuple([0.0] * NUM_JOINTS)

        if smbus2 is None:
            print("[WARN] smbus2 is not installed, using simulated joint state.")
            return

        try:
            self._bus = smbus2.SMBus(bus)
        except FileNotFoundError as exc:
            print(f"[WARN] Atmega I2C bus unavailable (/dev/i2c-{bus}), using simulated joint state: {exc}")
        except OSError as exc:
            print(f"[WARN] Failed to open Atmega I2C bus {bus}, using simulated joint state: {exc}")

    def send_command(self, angles: list[float]) -> None:
        with self._lock:
            self._pending_cmd = list(angles)

    def stop(self) -> None:
        self._stopping = True
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        self.stop()
        if self._bus is not None:
            self._bus.close()
