from threading import Lock

NUM_JOINTS = 6

try:
    import serial 
except ImportError:
    serial = None


class AtmegaI2C:
    """UART-based ATmega command sender."""

    def __init__(self, port: str = "/dev/ttyTHS1", baudrate: int = 115200, timeout: float = 0.1):
        self._lock = Lock()
        self._serial = None

        if serial is None:
            print("[WARN] pyserial is not installed, UART commands will be skipped.")
            return

        try:
            self._serial = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout,
                write_timeout=timeout,
            )
        except serial.SerialException as exc:
            print(f"[WARN] Failed to open UART port {port}: {exc}")
            self._serial = None

    def _format_command(self, angles) -> bytes:
        values = list(angles)[:NUM_JOINTS]
        if len(values) < NUM_JOINTS:
            values.extend([0.0] * (NUM_JOINTS - len(values)))
        # servos [0, 1] = joint position & stepper [2-5] number of steps, positive -> dir = 1, neg -> dir = 0 
        payload = ",".join(f"{v:.6f}" for v in values)
        return f"CMD,{payload}\n".encode("ascii")

    def send_command(self, angles) -> None:
        if self._serial is None:
            return

        message = self._format_command(angles)
        with self._lock:
            try:
                self._serial.write(message)
                self._serial.flush()
            except Exception as exc:
                print(f"[WARN] UART write failed: {exc}")
``
    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None