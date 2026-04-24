#!/usr/bin/env python3
import argparse
import math
import os
import serial
import statistics
import struct
import time
from threading import RLock
from typing import List, Optional


PORT = os.getenv("RS485_PORT", "/dev/ttyUSB0")
BAUDRATE = int(os.getenv("RS485_BAUDRATE", "256000"))
READ_RETRIES = int(os.getenv("RS485_READ_RETRIES", "1"))
# Change these values at the top of your script
TIMEOUT = float(os.getenv("RS485_TIMEOUT", "0.01"))             # Reduced from 0.1 to 10ms
TURNAROUND_DELAY_S = float(os.getenv("RS485_TURNAROUND_DELAY_S", "0.0")) # Set to 0.0
INTER_READ_DELAY_S = float(os.getenv("RS485_INTER_READ_DELAY_S", "0.0")) # Set to 0.0
PARITY = os.getenv("RS485_PARITY", "N").upper()
STOPBITS = float(os.getenv("RS485_STOPBITS", "1"))
ENDIAN = os.getenv("RS485_ENDIAN", ">")
DEBUG_TIMING = os.getenv("RS485_DEBUG_TIMING", "1") == "1"
COUNTS_PER_REV = 16384
PULSES_PER_REV = 200
RADIANS_PER_COUNT = 2 * math.pi / COUNTS_PER_REV
GEAR_RATIO = 5
ZERO_GRAVITY_POS = {1: -0.35, 2: -0.35, 3: -0.35, 4: -0.35}

class Rs485Driver:
    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        timeout: float = TIMEOUT,
        retries: int = READ_RETRIES,
        ser: Optional[serial.Serial] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retries = max(1, retries)
        self._serial_lock = RLock()
        self._send_priority = False
        self._last_error_log = {addr: 0.0 for addr in range(1, 5)}
        self.serial = ser or serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        ) 
        self._zero_position_counts: dict[int, Optional[int]] = {
            addr: None for addr in range(1, 5)
        }
        self._last_positions = {addr: 0.0 for addr in range(1, 5)}

    def __enter__(self) -> "Rs485Driver":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    @staticmethod
    def checksum(frame_wo_checksum: bytes) -> int:
        """
        MKS checksum used by firmware:
        sum of all frame bytes (header+addr+cmd+data...), low 8 bits.
        """
        return sum(frame_wo_checksum) & 0xFF

    @classmethod
    def append_crc(cls, frame: bytes) -> bytes:
        crc = cls.checksum(frame)
        return frame + bytes([crc])

    @classmethod
    def build_read_cmd(cls, addr: int) -> bytes:
        frame = bytearray([0xFA, addr & 0xFF, 0x30])
        frame.append(cls.checksum(frame))
        return bytes(frame)

    @staticmethod
    def read_exact(ser: serial.Serial, n: int) -> bytes:
        deadline = time.time() + 0.5
        buf = bytearray()
        while len(buf) < n and time.time() < deadline:
            chunk = ser.read(n - len(buf))
            if chunk:
                buf.extend(chunk)
        return bytes(buf)

    @classmethod
    def parse_encoder_response(cls, addr: int, resp: bytes) -> tuple[int, int]:
        if len(resp) != 10:
            raise ValueError(f"Bad response length: {len(resp)} bytes, got {resp.hex()}")

        if resp[0] != 0xFB:
            raise ValueError(f"Bad header from encoder {addr}: {resp[0]:02X}")

        if resp[1] != (addr & 0xFF):
            raise ValueError(f"Bad address in response from encoder {addr}: {resp.hex()}")

        if resp[2] != 0x30:
            raise ValueError(f"Bad function code from encoder {addr}: {resp[2]:02X}")

        calc_crc = cls.checksum(resp[:-1])
        rx_crc = resp[-1]
        if calc_crc != rx_crc:
            raise ValueError(
                f"CRC mismatch on encoder {addr}: expected 0x{calc_crc:02X}, got 0x{rx_crc:02X}"
            )

        carry = struct.unpack(f"{ENDIAN}i", resp[3:7])[0]
        value = struct.unpack(f"{ENDIAN}H", resp[7:9])[0]
        return carry, value

    def _read_encoder_once(self, addr: int) -> float:
        cmd = self.build_read_cmd(addr)
        lock_wait_start = time.time()
        while self._send_priority or not self._serial_lock.acquire(blocking=False):
            time.sleep(0.001)
        lock_acquired = time.time()
        try:
            write_start = time.time()
            self.serial.write(cmd)
            self.serial.flush()
            write_done = time.time()

            read_exact_start = time.time()
            resp = self.read_exact(self.serial, 10)
            read_done = time.time()
            if DEBUG_TIMING:
                print(
                    f"[ENC {addr}] lock_wait={lock_acquired - lock_wait_start:.4f}s  "
                    f"write={write_done - write_start:.4f}s  "
                    f"read_exact={read_done - read_exact_start:.4f}s  "
                    f"total={read_done - lock_acquired:.4f}s"
                )
            if len(resp) != 10:
                raise TimeoutError(f"Short response from encoder {addr}: {resp.hex()}")

            carry, value = self.parse_encoder_response(addr, resp)
        finally:
            self._serial_lock.release()
        position_counts = carry * COUNTS_PER_REV + value

        zero_counts = self._zero_position_counts[addr]
        if zero_counts is None:
            zero_counts = position_counts
            self._zero_position_counts[addr] = zero_counts

        relative_counts = position_counts - zero_counts
        radians = relative_counts * RADIANS_PER_COUNT / GEAR_RATIO
        if (addr == 0x01 or addr == 0x04):
            radians *= -1
        self._last_positions[addr] = radians
        return radians

    def read_encoder(self, addr: int) -> float:
        """
        Reads one encoder and returns a joint angle in radians.
        The first successful read for each encoder is treated as 0.0 rad.
        """
        
        read_start_time = time.time()
        for attempt in range(self.retries):
            try:
                return self._read_encoder_once(addr)
            except Exception as exc:
                if attempt < self.retries - 1 and INTER_READ_DELAY_S > 0:
                    time.sleep(INTER_READ_DELAY_S)
                    continue

                now = time.monotonic()
                # Throttle error spam if the state loop runs at high frequency.
                if now - self._last_error_log[addr] > 1.0:
                    print(
                        f"Error reading encoder {addr} on {self.port} @ {self.baudrate}: {exc}"
                    )
                    self._last_error_log[addr] = now
                return self._last_positions.get(addr, 0.0)
        read_end_time = time.time()
        print("read time: ", read_end_time - read_start_time)
        return self._last_positions.get(addr, 0.0)

    def _counts_to_radians(self, addr: int, carry: int, value: int) -> float:
        position_counts = carry * COUNTS_PER_REV + value

        zero_counts = self._zero_position_counts[addr]
        if zero_counts is None:
            zero_counts = position_counts
            self._zero_position_counts[addr] = zero_counts

        relative_counts = position_counts - zero_counts
        radians = relative_counts * RADIANS_PER_COUNT / GEAR_RATIO
        if addr in (0x01, 0x04):
            radians *= -1
        self._last_positions[addr] = radians
        return radians

    def _parse_burst_encoder_responses(self, raw: bytes, expected_addrs: set[int]) -> dict[int, float]:
        found: dict[int, float] = {}
        i = 0
        max_i = len(raw) - 10

        while i <= max_i and len(found) < len(expected_addrs):
            if raw[i] != 0xFB:
                i += 1
                continue

            frame = raw[i : i + 10]
            if len(frame) < 10:
                break

            addr = frame[1]
            if frame[2] != 0x30 or addr not in expected_addrs:
                i += 1
                continue

            try:
                carry, value = self.parse_encoder_response(addr, frame)
            except Exception:
                i += 1
                continue

            found[addr] = self._counts_to_radians(addr, carry, value)
            i += 10

        return found

    def read_all_joints_serial(self) -> dict:
        joints: List[float] = []
        for addr in range(1, 5):
            joints.append(self.read_encoder(addr))
        return {"joints": joints}
    
    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.serial.read(n - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _send_frame(self, frame: bytes, expect_len: int, check_ack: bool = True) -> bytes:
        frame = self.append_crc(frame)
        with self._serial_lock:
            self.serial.write(frame)
            self.serial.flush()

            if check_ack: 
                return self._read_exact(expect_len)
            else:
                # returns a mock ACK with status set to 1
                return b"\x00\x00\x00\x01"
    def read_all_joints_batch(self) -> dict:
        """
        Burst-send all 4 encoder query frames, then read all 40 bytes back
        in one shot. Eliminates per-encoder lock/flush/write overhead.
        """
        addrs = [1, 2, 3, 4]
        expected = set(addrs)
        burst = b"".join(self.build_read_cmd(a) for a in addrs)  # 16 bytes total

        with self._serial_lock:
            self.serial.write(burst)
            self.serial.flush()
            if TURNAROUND_DELAY_S > 0:
                time.sleep(TURNAROUND_DELAY_S)
            raw = self.read_exact(self.serial, 40)  # 10 bytes × 4 encoders

        parsed = self._parse_burst_encoder_responses(raw, expected)
        joints = []
        for addr in addrs:
            if addr in parsed:
                joints.append(parsed[addr])
            else:
                now = time.monotonic()
                if now - self._last_error_log[addr] > 1.0:
                    print(f"Missing/invalid burst frame for encoder {addr}: {raw.hex()}")
                    self._last_error_log[addr] = now
                joints.append(self._last_positions.get(addr, 0.0))

        return {"joints": joints}

    def read_all_joints(self) -> dict:
        """Default path used by the rest of the code: batch read."""
        return self.read_all_joints_batch()
    def move_position(
        self,
        addr: int,
        pulses: int,
        speed: int = 0x0280,
        acc: int = 2,
        direction: int = 0,
        check_ack: bool = False,
    ) -> int:
        """
        Manual 6.4:
          Downlink: FA addr FD [byte4] [byte5] acc pulses(4) CRC
          Uplink:   FB addr FD status CRC
          status = 0 fail
          status = 1 starting
          status = 2 complete

        byte4:
          bit7 = direction
          bits3:0 + byte5 = speed
        """
        if not (0 <= speed <= 0x0FFF):
            raise ValueError("speed must be 0..4095")
        if not (0 <= acc <= 32):
            raise ValueError("acc must be 0..32")
        if not (0 <= pulses <= 0xFFFFFFFF):
            raise ValueError("pulses must be 0..0xFFFFFFFF")
        if direction not in (0, 1):
            raise ValueError("direction must be 0 (CW) or 1 (CCW)")

        byte4 = ((direction & 0x01) << 7) | ((speed >> 8) & 0x0F)
        byte5 = speed & 0xFF
        pulse_bytes = int(pulses).to_bytes(4, byteorder="big", signed=False)

        frame = bytes([0xFA, addr & 0xFF, 0xFD, byte4, byte5, acc & 0xFF]) + pulse_bytes
        resp = self._send_frame(frame, 4, check_ack=check_ack)

        # if len(resp) != 4:
        #     raise TimeoutError(f"Short move response from motor {addr}: {resp.hex()}")

        # if resp[0] != 0xFB or resp[1] != (addr & 0xFF) or resp[2] != 0xFD:
        #     raise ValueError(f"Bad move response header from motor {addr}: {resp.hex()}")

        return resp[3]

    def send_joint_position(self, joints: dict, max_pulses: int = 20):
        """
        Expected input examples:
          {1: 0.1, 2: -0.2, 3: 1.57, 4: 0.0}
        where values are absolute joint targets in radians.

        Each target is converted into a delta from the last cached encoder
        position, then sent as a motor move command.
        """
        results = {}
        addr_times = {}
        self._send_priority = True

        for addr, target_rad in joints.items():
            if not isinstance(addr, int):
                continue

            current_rad = self._last_positions.get(addr, 0.0)
            delta_rad = target_rad - current_rad
            pulses = int(round((abs(delta_rad) / (2 * math.pi)) * PULSES_PER_REV) * GEAR_RATIO)
            pulses = min(pulses, max_pulses)
            if (addr == 0x01 or addr == 0x04):
                direction = 1 if delta_rad <= 0 else 0
            else:
                direction = 1 if delta_rad >= 0 else 0
            # input("Move one!")
            t0 = time.time()
            status = self.move_position(
                addr=addr,
                pulses=pulses,
                speed=10,
                acc=2,
                direction=direction,
            )
            addr_times[addr] = time.time() - t0
            results[addr] = {
                "enabled": True,
                "status": status,
                "pulses": pulses,
            }
        self._send_priority = False
        times_str = "  ".join(f"addr{a}={t:.4f}s" for a, t in addr_times.items())
        avg = sum(addr_times.values()) / len(addr_times) if addr_times else 0
        print(f"[SEND] {times_str}  avg={avg:.4f}s  total={sum(addr_times.values()):.4f}s")
        return results

    def shutdown(self, tolerance: float = 0.01, step_delay: float = 0.05):
        """Gradually move all joints to ZERO_GRAVITY_POS before power off."""
        print("[SHUTDOWN] Moving to zero gravity position...")
        while True:
            self.send_joint_position(ZERO_GRAVITY_POS, max_pulses=10)
            if all(
                abs(self._last_positions.get(addr, 0.0) - target) < tolerance
                for addr, target in ZERO_GRAVITY_POS.items()
            ):
                break
            time.sleep(step_delay)
        print("[SHUTDOWN] Zero gravity position reached.")


def _summarize_times(label: str, samples: List[float]) -> None:
    if not samples:
        print(f"{label}: no samples")
        return

    mean_s = statistics.mean(samples)
    median_s = statistics.median(samples)
    min_s = min(samples)
    max_s = max(samples)
    hz = (1.0 / mean_s) if mean_s > 0 else 0.0
    print(
        f"{label}: n={len(samples)}  mean={mean_s*1000:.3f} ms  "
        f"median={median_s*1000:.3f} ms  min={min_s*1000:.3f} ms  "
        f"max={max_s*1000:.3f} ms  rate={hz:.2f} Hz"
    )


def _benchmark_mode(
    driver: Rs485Driver,
    mode: str,
    iterations: int,
    warmup: int,
    delay_s: float,
) -> List[float]:
    fn = driver.read_all_joints_batch if mode == "batch" else driver.read_all_joints_serial

    for _ in range(max(0, warmup)):
        fn()
        if delay_s > 0:
            time.sleep(delay_s)

    samples: List[float] = []
    for _ in range(max(1, iterations)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
        if delay_s > 0:
            time.sleep(delay_s)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RS485 encoder reads: serial-per-encoder vs batch burst"
    )
    parser.add_argument(
        "--mode",
        choices=["serial", "batch", "both"],
        default="both",
        help="Which read method to benchmark",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Measured iterations per mode",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Unmeasured warmup iterations per mode",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Sleep between iterations in seconds",
    )
    args = parser.parse_args()

    with Rs485Driver() as driver:
        if args.mode in ("serial", "both"):
            serial_samples = _benchmark_mode(
                driver,
                mode="serial",
                iterations=args.iterations,
                warmup=args.warmup,
                delay_s=args.delay,
            )
            _summarize_times("serial", serial_samples)
        else:
            serial_samples = []

        if args.mode in ("batch", "both"):
            batch_samples = _benchmark_mode(
                driver,
                mode="batch",
                iterations=args.iterations,
                warmup=args.warmup,
                delay_s=args.delay,
            )
            _summarize_times("batch", batch_samples)
        else:
            batch_samples = []

    if serial_samples and batch_samples:
        serial_mean = statistics.mean(serial_samples)
        batch_mean = statistics.mean(batch_samples)
        if batch_mean > 0:
            speedup = serial_mean / batch_mean
            print(f"speedup (serial/batch): {speedup:.2f}x")


if __name__ == "__main__":
    main()
