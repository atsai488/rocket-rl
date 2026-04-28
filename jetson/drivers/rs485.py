#!/usr/bin/env python3
import math
import os
import serial
import struct
import time
from threading import RLock
from typing import List, Optional


PORT = os.getenv("RS485_PORT", "/dev/ttyUSB0")
BAUDRATE = int(os.getenv("RS485_BAUDRATE", "256000"))
TIMEOUT = float(os.getenv("RS485_TIMEOUT", "0.01"))
READ_RETRIES = int(os.getenv("RS485_READ_RETRIES", "1"))
TURNAROUND_DELAY_S = float(os.getenv("RS485_TURNAROUND_DELAY_S", "0.0"))
INTER_READ_DELAY_S = float(os.getenv("RS485_INTER_READ_DELAY_S", "0.0"))
INTER_MOVE_FRAME_DELAY_S = float(os.getenv("RS485_INTER_MOVE_FRAME_DELAY_S", "0.001"))
PARITY = os.getenv("RS485_PARITY", "N").upper()
STOPBITS = float(os.getenv("RS485_STOPBITS", "1"))
ENDIAN = os.getenv("RS485_ENDIAN", ">")
DEBUG_TIMING = os.getenv("RS485_DEBUG_TIMING", "0") == "1"
COUNTS_PER_REV = 16384
PULSES_PER_REV = 200
RADIANS_PER_COUNT = 2 * math.pi / COUNTS_PER_REV
GEAR_RATIO = 5
ZERO_GRAVITY_POS = {1: -0.35, 2: -0.35, 3: -0.35, 4: -0.35}
HIP_ADDRS = {1, 2}   # Revolute3, Revolute4
KNEE_ADDRS = {3, 4}  # Revolute5, Revolute6

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

    def read_all_joints(self) -> dict:
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

    def send_joint_position(self, joints: dict, hip_pulses: int = 10, knee_pulses: int = 20):
        """
        Expected input examples:
          {1: 0.1, 2: -0.2, 3: 1.57, 4: 0.0}
        where values are absolute joint targets in radians.

        Each target is converted into a delta from the last cached encoder
        position, then sent as a motor move command.
        """
        results = {}
        addr_times = {}
        prepared_frames: list[tuple[int, bytes]] = []
        self._send_priority = True

        try:
            for addr, target_rad in joints.items():
                if not isinstance(addr, int):
                    continue

                current_rad = self._last_positions.get(addr, 0.0)
                delta_rad = target_rad - current_rad
                pulses = int(round((abs(delta_rad) / (2 * math.pi)) * PULSES_PER_REV) * GEAR_RATIO)
                max_p = hip_pulses if addr in HIP_ADDRS else knee_pulses
                pulses = min(pulses, max_p)
                if (addr == 0x01 or addr == 0x04):
                    direction = 1 if delta_rad <= 0 else 0
                else:
                    direction = 1 if delta_rad >= 0 else 0

                byte4 = ((direction & 0x01) << 7) | ((10 >> 8) & 0x0F)
                byte5 = 10 & 0xFF
                pulse_bytes = int(pulses).to_bytes(4, byteorder="big", signed=False)
                frame = (
                    bytes([0xFA, addr & 0xFF, 0xFD, byte4, byte5, 2 & 0xFF])
                    + pulse_bytes
                )
                prepared_frames.append((addr, self.append_crc(frame)))

                # No ACK mode: keep status behavior compatible with previous mock ACK path.
                addr_times[addr] = 0.0
                results[addr] = {
                    "enabled": True,
                    "status": 1,
                    "pulses": pulses,
                }

            if prepared_frames:
                with self._serial_lock:
                    for idx, (addr, frame) in enumerate(prepared_frames):
                        t0 = time.time()
                        self.serial.write(frame)
                        self.serial.flush()
                        addr_times[addr] = time.time() - t0

                        # Some motor controllers do not reliably parse back-to-back
                        # frames without a tiny spacing gap.
                        if idx < len(prepared_frames) - 1 and INTER_MOVE_FRAME_DELAY_S > 0:
                            time.sleep(INTER_MOVE_FRAME_DELAY_S)
        finally:
            self._send_priority = False

        times_str = "  ".join(f"addr{a}={t:.4f}s" for a, t in addr_times.items())
        avg = sum(addr_times.values()) / len(addr_times) if addr_times else 0
        print(f"[SEND] {times_str}  avg={avg:.4f}s  total={sum(addr_times.values()):.4f}s")
        return results

    def send_joint_position_fast(self, joints: dict, hip_pulses: int = 10, knee_pulses: int = 20):
        """
        Same as send_joint_position but batches all 4 frames into a single
        serial.write() + serial.flush() to minimise syscall overhead on the
        Jetson's tegra-uart DMA / USB-serial FIFO.  Use when
        INTER_MOVE_FRAME_DELAY_S == 0 and the motor controllers can handle
        back-to-back frames without a gap.
        """
        results = {}
        prepared_frames: list[tuple[int, bytes]] = []
        self._send_priority = True

        try:
            for addr, target_rad in joints.items():
                if not isinstance(addr, int):
                    continue

                current_rad = self._last_positions.get(addr, 0.0)
                delta_rad = target_rad - current_rad
                pulses = int(round((abs(delta_rad) / (2 * math.pi)) * PULSES_PER_REV) * GEAR_RATIO)
                max_p = hip_pulses if addr in HIP_ADDRS else knee_pulses
                pulses = min(pulses, max_p)
                if addr == 0x01 or addr == 0x04:
                    direction = 1 if delta_rad <= 0 else 0
                else:
                    direction = 1 if delta_rad >= 0 else 0

                byte4 = ((direction & 0x01) << 7) | ((10 >> 8) & 0x0F)
                byte5 = 10 & 0xFF
                pulse_bytes = int(pulses).to_bytes(4, byteorder="big", signed=False)
                frame = (
                    bytes([0xFA, addr & 0xFF, 0xFD, byte4, byte5, 2 & 0xFF])
                    + pulse_bytes
                )
                prepared_frames.append((addr, self.append_crc(frame)))
                results[addr] = {"enabled": True, "status": 1, "pulses": pulses}

            if prepared_frames:
                if DEBUG_TIMING:
                    t0 = time.time()
                with self._serial_lock:
                    for idx, (_, frame) in enumerate(prepared_frames):
                        self.serial.write(frame)
                        if idx < len(prepared_frames) - 1 and INTER_MOVE_FRAME_DELAY_S > 0:
                            time.sleep(INTER_MOVE_FRAME_DELAY_S)
                    self.serial.flush()
                if DEBUG_TIMING:
                    print(f"[SEND_FAST] {len(prepared_frames)} frames in {time.time() - t0:.4f}s")
        finally:
            self._send_priority = False

        pulses_str = "  ".join(f"addr{addr}={r['pulses']}p" for addr, r in results.items())
        print(f"[PULSES] {pulses_str}")
        return results

    def shutdown(self, tolerance: float = 0.01, step_delay: float = 0.05):
        """Gradually move all joints to ZERO_GRAVITY_POS before power off."""
        print("[SHUTDOWN] Moving to zero gravity position...")
        while True:
            self.send_joint_position(ZERO_GRAVITY_POS, hip_pulses=10, knee_pulses=20)
            if all(
                abs(self._last_positions.get(addr, 0.0) - target) < tolerance
                for addr, target in ZERO_GRAVITY_POS.items()
            ):
                break
            time.sleep(step_delay)
        print("[SHUTDOWN] Zero gravity position reached.")
