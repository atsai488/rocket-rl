from threading import Lock

NUM_JOINTS = 6

try:
    import serial  # type: ignore[import-not-found]
except ImportError:
    serial = None


class AtmegaI2C:
    """UART-based ATmega command sender.

    The class name is kept for compatibility with the existing imports, but the
    implementation now sends command messages over the Jetson Nano UART header.
    On the Nano, that is typically exposed as /dev/ttyTHS1.
    """

    def __init__(self, port: str = "/dev/ttyTHS1", baudrate: int = 115200, timeout: float = 0.1):
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial = None
        self._pending_cmd = None
        self._lock = Lock()
        self._thread = None
        self._stopping = False
        self._fallback_state = tuple([0.0] * NUM_JOINTS)

        if serial is None:
            print("[WARN] pyserial is not installed, UART commands will be skipped.")
            return

        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=self._timeout)
        except Exception as exc:
            print(f"[WARN] Failed to open UART port {self._port}: {exc}")
            self._serial = None

    def _format_command(self, angles: list[float]) -> bytes:
        values = list(angles)[:NUM_JOINTS]
        if len(values) < NUM_JOINTS:
            values.extend([0.0] * (NUM_JOINTS - len(values)))
        payload = ",".join(f"{value:.6f}" for value in values)
        return f"CMD,{payload}\n".encode("ascii")

    def send_command(self, angles: list[float]) -> None:
        message = self._format_command(angles)

        with self._lock:
            self._pending_cmd = list(angles)

            if self._serial is None:
                return

            try:
                self._serial.write(message)
                self._serial.flush()
            except Exception as exc:
                print(f"[WARN] UART write failed: {exc}")

    def stop(self) -> None:
        self._stopping = True
        if self._thread is not None:
            self._thread.join()

    def close(self) -> None:
        self.stop()
        if self._serial is not None:
            self._serial.close()
