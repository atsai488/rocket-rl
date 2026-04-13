#!/usr/bin/env python3
import math
import serial
import struct
from typing import List, Optional


PORT = "/dev/ttyUSB0"      # change to /dev/ttyTHS1 if using Jetson UART
BAUDRATE = 38400          # change to your encoder's baud rate
TIMEOUT = 0.2
ENDIAN = "<"               # "<" little-endian, ">" big-endian
COUNTS_PER_REV = 16384
RADIANS_PER_COUNT = 2 * math.pi / COUNTS_PER_REV
GEAR_RATIO = 5

class Rs485Driver:
    def __init__(
        self,
        port: str = PORT,
        baudrate: int = BAUDRATE,
        timeout: float = TIMEOUT,
        ser: Optional[serial.Serial] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = ser or serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self._last_positions = {addr: 0.0 for addr in range(1, 5)}

    def __enter__(self) -> "Rs485Driver":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    @staticmethod
    def crc16_modbus(data: bytes) -> int:
        """
        Standard Modbus CRC-16.
        Returns integer CRC; append as low byte then high byte.
        """
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    @classmethod
    def append_crc(cls, frame: bytes) -> bytes:
        crc = cls.crc16_modbus(frame)
        return frame + struct.pack("<H", crc)

    @classmethod
    def build_read_cmd(cls, addr: int) -> bytes:
        frame = bytes([0xFA, addr & 0xFF, 0x30])
        return cls.append_crc(frame)

    @staticmethod
    def read_exact(ser: serial.Serial, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = ser.read(n - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def read_encoder(self, addr: int) -> float:
        """
        Reads one encoder and returns a joint angle in radians.
        The first successful read for each encoder is treated as 0.0 rad.
        """
        try:
            cmd = self.build_read_cmd(addr)
            self.serial.reset_input_buffer()
            self.serial.write(cmd)
            self.serial.flush()

            # Expected response:
            # FB addr 30 carry(4) value(2) crc(2) => 11 bytes total
            resp = self.read_exact(self.serial, 11)
            if len(resp) != 11:
                raise TimeoutError(f"Short response from encoder {addr}: {resp.hex()}")

            # Basic header check
            if resp[0] != 0xFB or resp[1] != (addr & 0xFF) or resp[2] != 0x30:
                raise ValueError(f"Bad response header from encoder {addr}: {resp.hex()}")

            # CRC check
            payload = resp[:-2]
            rx_crc = struct.unpack("<H", resp[-2:])[0]
            calc_crc = self.crc16_modbus(payload)
            if rx_crc != calc_crc:
                raise ValueError(
                    f"CRC mismatch on encoder {addr}: rx=0x{rx_crc:04X}, calc=0x{calc_crc:04X}"
                )

            data = resp[3:-2]
            carry, value = struct.unpack(f"{ENDIAN}iH", data)
            position_counts = carry * COUNTS_PER_REV + value
            
            radians = position_counts * RADIANS_PER_COUNT / GEAR_RATIO
            self._last_positions[addr] = radians
            return radians
        except Exception as exc:
            print(f"Error reading encoder {addr}: {exc}")
            return self._last_positions.get(addr, 0)

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
        self.serial.write(frame)
        self.serial.flush()
        return self._read_exact(expect_len)

    def enable_motor(self, addr: int, enable: bool = True) -> bool:
        """
        Manual 6.2:
          Downlink: FA addr F3 en CRC
          Uplink:   FB addr F3 status CRC
          status = 1 => success
          status = 0 => fail
        """
        frame = bytes([0xFA, addr & 0xFF, 0xF3, 0x01 if enable else 0x00])
        resp = self._send_frame(frame, 6)

        if len(resp) != 6:
            raise TimeoutError(f"Short enable response from motor {addr}: {resp.hex()}")

        if resp[0] != 0xFB or resp[1] != (addr & 0xFF) or resp[2] != 0xF3:
            raise ValueError(f"Bad enable response header from motor {addr}: {resp.hex()}")

        payload = resp[:-2]
        rx_crc = struct.unpack("<H", resp[-2:])[0]
        calc_crc = self.crc16_modbus(payload)
        if rx_crc != calc_crc:
            raise ValueError(f"CRC mismatch on enable response motor {addr}")

        status = resp[3]
        return status == 1

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
        resp = self._send_frame(frame, 6)

        if len(resp) != 6:
            raise TimeoutError(f"Short move response from motor {addr}: {resp.hex()}")

        if resp[0] != 0xFB or resp[1] != (addr & 0xFF) or resp[2] != 0xFD:
            raise ValueError(f"Bad move response header from motor {addr}: {resp.hex()}")

        payload = resp[:-2]
        rx_crc = struct.unpack("<H", resp[-2:])[0]
        calc_crc = self.crc16_modbus(payload)
        if rx_crc != calc_crc:
            raise ValueError(f"CRC mismatch on move response motor {addr}")

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

            if not self.enable_motor(addr, True):
                results[addr] = {"enabled": False, "status": None}
                continue

            current_rad = self._last_positions.get(addr, 0.0)
            delta_rad = target_rad - current_rad
            pulses = int(round((abs(delta_rad) / (2 * math.pi)) * COUNTS_PER_REV) * GEAR_RATIO)
            direction = 0 if delta_rad >= 0 else 1

            status = self.move_position(
                addr=addr,
                pulses=pulses,
                speed=0x0280,
                acc=2,
                direction=direction,
            )
            results[addr] = {
                "enabled": True,
                "status": status,
                "pulses": pulses,
            }

        return results