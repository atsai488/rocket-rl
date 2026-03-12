import struct
import time
from threading import Thread, Lock
from typing import Callable
import smbus2

ATMEGA_ADDR = 0x08  # i2c address of the atmega
REG_MOTOR_STATE = (
    0x00  # starting register for motor state (6 joints * 4 bytes each = 24 bytes)
)
REG_JOINT_CMD = (
    0x01  # starting register for joint command (6 joints * 4 bytes each = 24 bytes)
)
NUM_JOINTS = 6
STATE_BYTES = NUM_JOINTS * 4


class AtmegaI2C:
    def __init__(
        self, bus: int = 1, addr: int = ATMEGA_ADDR, poll_interval: float = 0.01
    ):
        self._bus = smbus2.SMBus(bus)
        self._addr = addr
        self._poll_interval = poll_interval
        self._pending_cmd = None
        self._lock = Lock()
        self._thread = None
        self._stopping = False

    def start(self, on_state_update: Callable[[dict], None]) -> None:        
        self._stopping = False
        self._thread = Thread(
            target=self._poll_loop, args=(on_state_update,), daemon=True
        )
        self._thread.start()

    def send_command(self, angles: list[float]) -> None:
        with self._lock:
            self._pending_cmd = list(angles)

    def stop(self) -> None:
        self._stopping = True
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        self.stop()
        self._bus.close()

    def _poll_loop(self, on_state_update: Callable[[dict], None]) -> None:
        while not self._stopping:
            cycle_start = time.perf_counter()

            self._read_and_dispatch(on_state_update)
            self._flush_command()

            elapsed = time.perf_counter() - cycle_start
            remaining = self._poll_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _read_and_dispatch(self, on_state_update: Callable[[dict], None]) -> None:
        try:
            raw = self._bus.read_i2c_block_data(
                self._addr, REG_MOTOR_STATE, STATE_BYTES
            )
            angles = struct.unpack("<6f", bytes(raw))
            on_state_update({"joints": angles})
        except OSError as e:
            print(f"[AtmegaI2C] read error: {e}")

    def _flush_command(self) -> None:
        with self._lock:
            cmd = self._pending_cmd
            self._pending_cmd = None

        if cmd is None:
            return

        try:
            payload = list(struct.pack("<6f", *cmd))
            self._bus.write_i2c_block_data(self._addr, REG_JOINT_CMD, payload)
        except OSError as e:
            print(f"[AtmegaI2C] write error: {e}")
