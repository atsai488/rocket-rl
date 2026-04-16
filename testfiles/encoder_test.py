import os
import time
import struct
import serial

PORT = os.getenv("RS485_PORT", "/dev/ttyUSB0")
BAUDRATE = int(os.getenv("RS485_BAUDRATE", "38400"))

ADDR = 0x01
ENCODER_ADDRS = [1, 2, 3, 4]
FUNC_READ_ENCODER = 0x30
FUNC_ENABLE_MOTOR = 0xF3
FUNC_MOVE_POSITION = 0xFD
COUNTS_PER_REV = 0x4000


def calc_crc(data: bytes) -> int:
    return sum(data) & 0xFF


def build_read_command(addr: int = ADDR) -> bytes:
    frame = bytearray([0xFA, addr, FUNC_READ_ENCODER])
    frame.append(calc_crc(frame))
    return bytes(frame)


def read_exactly(ser: serial.Serial, n: int, timeout_s: float = 0.5) -> bytes:
    deadline = time.time() + timeout_s
    buf = bytearray()

    while len(buf) < n and time.time() < deadline:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf.extend(chunk)

    return bytes(buf)


def parse_encoder_response(resp: bytes):
    # Expected: FB addr 30 carry(4) value(2) crc(1) => 10 bytes total
    if len(resp) != 10:
        raise ValueError(f"Bad response length: {len(resp)} bytes, got {resp.hex()}")

    if resp[0] != 0xFB:
        raise ValueError(f"Bad header: {resp[0]:02X}")

    if resp[2] != FUNC_READ_ENCODER:
        raise ValueError(f"Bad function code: {resp[2]:02X}")

    crc_expected = calc_crc(resp[:-1])
    crc_received = resp[-1]
    if crc_expected != crc_received:
        raise ValueError(f"CRC mismatch: expected {crc_expected:02X}, got {crc_received:02X}")
    

    # Encoder payload is MSB-first (big-endian): carry(4), value(2)
    carry = struct.unpack(">i", resp[3:7])[0]
    value = struct.unpack(">H", resp[7:9])[0]

    return carry, value


def read_encoder(ser: serial.Serial, addr: int = ADDR):
    ser.reset_input_buffer()
    cmd = build_read_command(addr)
    ser.write(cmd)
    ser.flush()

    resp = read_exactly(ser, 10, timeout_s=0.5)
    print(f"RESP addr={addr}:", resp.hex())
    if len(resp) != 10:
        raise TimeoutError(f"Incomplete response: {resp.hex()}")
    if resp[1] != (addr & 0xFF):
        raise ValueError(f"Response address mismatch: expected {addr}, got frame {resp.hex()}")

    return parse_encoder_response(resp)


def send_and_read_status(ser: serial.Serial, frame: bytes, timeout_s: float = 0.5) -> int:
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()

    resp = read_exactly(ser, 4, timeout_s=timeout_s)
    if len(resp) != 4:
        raise TimeoutError(f"Incomplete status response: {resp.hex()}")
    if resp[0] != 0xFB:
        raise ValueError(f"Bad status header: {resp.hex()}")
    if resp[1] != ADDR:
        raise ValueError(f"Bad status address: {resp.hex()}")

    return resp[3]


def build_enable_command(addr: int = ADDR, enable: bool = True) -> bytes:
    frame = bytearray([0xFA, addr, FUNC_ENABLE_MOTOR, 0x01 if enable else 0x00])
    frame.append(calc_crc(frame))
    return bytes(frame)


def build_move_command(
    addr: int = ADDR,
    pulses: int = 50,
    speed: int = 0x0280,
    acc: int = 2,
    direction: int = 0,
) -> bytes:
    if not (0 <= speed <= 0x0FFF):
        raise ValueError("speed must be in [0, 4095]")
    if direction not in (0, 1):
        raise ValueError("direction must be 0 (forward/CW) or 1 (reverse/CCW)")

    byte4 = ((direction & 0x01) << 7) | ((speed >> 8) & 0x0F)
    byte5 = speed & 0xFF
    pulse_bytes = int(pulses).to_bytes(4, byteorder="big", signed=False)

    frame = bytearray([0xFA, addr, FUNC_MOVE_POSITION, byte4, byte5, acc & 0xFF])
    frame.extend(pulse_bytes)
    frame.append(calc_crc(frame))
    return bytes(frame)


def build_stop_command(addr: int = ADDR, acc: int = 2) -> bytes:
    frame = bytearray([0xFA, addr, FUNC_MOVE_POSITION, 0x00, 0x00, acc & 0xFF, 0x00, 0x00, 0x00, 0x00])
    frame.append(calc_crc(frame))
    return bytes(frame)


def main():
    zero_offsets = {}
    target_addr = 1

    with serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.05,
    ) as ser:
        try:
            enable_status = send_and_read_status(ser, build_enable_command(addr=target_addr, enable=True))
            move_status = send_and_read_status(ser, build_move_command(addr=target_addr, speed=1))
            print(f"Servo {target_addr}: enabled (status=0x{enable_status:02X}), move command sent at speed=1 (status=0x{move_status:02X})")
            time.sleep(0.5)
            move_status = send_and_read_status(ser, build_stop_command(addr=target_addr))
        except Exception as e:
            print(f"Servo {target_addr} command error: {e}")

        while True:
            for addr in ENCODER_ADDRS:
                try:
                    carry, value = read_encoder(ser, addr)
                    full_position = carry * COUNTS_PER_REV + value

                    if addr not in zero_offsets:
                        zero_offsets[addr] = full_position
                        print(f"Encoder {addr}: zero offset captured: {zero_offsets[addr]}")

                    relative_counts = full_position - zero_offsets[addr]
                    angle_deg = (relative_counts / COUNTS_PER_REV) * 360.0
                    print(
                        f"Encoder {addr}: carry={carry} value={value} full_position={full_position} "
                        f"relative_counts={relative_counts} angle_deg={angle_deg:.3f}"
                    )
                except Exception as e:
                    print(f"Encoder {addr} error: {e}")

            time.sleep(0.3)


if __name__ == "__main__":
    main()