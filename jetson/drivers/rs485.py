#!/usr/bin/env python3
import math
import os
import serial
import struct
import time
from typing import List, Optional


PORT = os.getenv("RS485_PORT", "/dev/ttyUSB0")
BAUDRATE = int(os.getenv("RS485_BAUDRATE", "256000"))
TIMEOUT = float(os.getenv("RS485_TIMEOUT", "0.1"))
READ_RETRIES = int(os.getenv("RS485_READ_RETRIES", "1"))
TURNAROUND_DELAY_S = float(os.getenv("RS485_TURNAROUND_DELAY_S", "0.0003"))
INTER_READ_DELAY_S = float(os.getenv("RS485_INTER_READ_DELAY_S", "0.0002"))
PARITY = os.getenv("RS485_PARITY", "N").upper()
STOPBITS = float(os.getenv("RS485_STOPBITS", "1"))
ENDIAN = os.getenv("RS485_ENDIAN", ">")
COUNTS_PER_REV = 16384
PULSES_PER_REV = 200
RADIANS_PER_COUNT = 2 * math.pi / COUNTS_PER_REV
GEAR_RATIO = 5

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
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.serial.write(cmd)
        self.serial.flush()
        # Some RS485 adapters need a short TX->RX turnaround window.
        if TURNAROUND_DELAY_S > 0:
            time.sleep(TURNAROUND_DELAY_S)

        resp = self.read_exact(self.serial, 10)
        if len(resp) != 10:
            raise TimeoutError(f"Short response from encoder {addr}: {resp.hex()}")

        carry, value = self.parse_encoder_response(addr, resp)
        position_counts = carry * COUNTS_PER_REV + value

        zero_counts = self._zero_position_counts[addr]
        if zero_counts is None:
            zero_counts = position_counts
            self._zero_position_counts[addr] = zero_counts

        relative_counts = position_counts - zero_counts
        radians = relative_counts * RADIANS_PER_COUNT / GEAR_RATIO
        self._last_positions[addr] = radians
        return radians

    def read_encoder(self, addr: int) -> float:
        """
        Reads one encoder and returns a joint angle in radians.
        The first successful read for each encoder is treated as 0.0 rad.
        """
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

    def _send_frame(self, frame: bytes, expect_len: int) -> bytes:
        frame = self.append_crc(frame)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()
        self.serial.write(frame)
        self.serial.flush()
        if TURNAROUND_DELAY_S > 0:
            time.sleep(TURNAROUND_DELAY_S)
        return self._read_exact(expect_len)


    def move_position(
        self,
        addr: int,
        pulses: int,
        speed: int = 0x0280,
        acc: int = 2,
        direction: int = 0,
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
        resp = self._send_frame(frame, 4)

        # if len(resp) != 4:
        #     raise TimeoutError(f"Short move response from motor {addr}: {resp.hex()}")

        # if resp[0] != 0xFB or resp[1] != (addr & 0xFF) or resp[2] != 0xFD:
        #     raise ValueError(f"Bad move response header from motor {addr}: {resp.hex()}")

        return resp[3]

    def send_joint_position(self, joints: dict):
        """
        Expected input examples:
          {1: 0.1, 2: -0.2, 3: 1.57, 4: 0.0}
        where values are absolute joint targets in radians.

        Each target is converted into a delta from the last cached encoder
        position, then sent as a motor move command.
        """
        results = {}

        for addr, target_rad in joints.items():
            if not isinstance(addr, int):
                continue

            current_rad = self._last_positions.get(addr, 0.0)
            delta_rad = target_rad - current_rad
            pulses = int(round((abs(delta_rad) / (2 * math.pi)) * PULSES_PER_REV) * GEAR_RATIO)
            if (addr == 0x01 or addr == 0x04):
                direction = 0 if delta_rad <= 0 else 1
            else:
                direction = 0 if delta_rad >= 0 else 1
            print(f"{addr}: delta rad: {delta_rad}, pulses: {pulses}", end="\n")
            # input("Move one!")
            status = self.move_position(
                addr=addr,
                pulses=pulses,
                speed=10,
                acc=2,
                direction=direction,
            )
            results[addr] = {
                "enabled": True,
                "status": status,
                "pulses": pulses,
            }
        time.sleep(5)
        print("\n")
        return results