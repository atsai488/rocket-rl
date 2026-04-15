import os
import time
import struct
import serial

PORT = os.getenv("RS485_PORT", "/dev/ttyUSB0")
BAUDRATE = int(os.getenv("RS485_BAUDRATE", "38400"))

ADDR = 0x01
FUNC_READ_ENCODER = 0x30


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
    

    # Most likely big-endian based on the protocol style
    carry = struct.unpack("<i", resp[3:7])[0]
    value = struct.unpack("<H", resp[7:9])[0]

    return carry, value


def read_encoder(ser: serial.Serial, addr: int = ADDR):
    ser.reset_input_buffer()
    cmd = build_read_command(addr)
    ser.write(cmd)
    ser.flush()

    resp = read_exactly(ser, 10, timeout_s=0.5)
    print("RESP:", resp.hex())
    if len(resp) != 10:
        raise TimeoutError(f"Incomplete response: {resp.hex()}")

    return parse_encoder_response(resp)


def main():
    with serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.05,
    ) as ser:
        while True:
            try:
                carry, value = read_encoder(ser)
                full_position = carry * 0x4000 + value
                print(f"carry={carry} value={value} full_position={full_position}")
            except Exception as e:
                print("Error:", e)

            time.sleep(0.1)


if __name__ == "__main__":
    main()