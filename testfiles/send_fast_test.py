import math
import os
import time
import serial
from drivers.rs485 import Rs485Driver, PULSES_PER_REV, GEAR_RATIO

PORT = os.getenv("RS485_PORT", "/dev/ttyUSB0")
BAUDRATE = int(os.getenv("RS485_BAUDRATE", "256000"))

PULSES_BACK = 10
DELTA_RAD = (PULSES_BACK / (PULSES_PER_REV * GEAR_RATIO)) * 2 * math.pi


def main():
    with serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.05,
    ) as ser:
        driver = Rs485Driver(ser=ser)

        pos3 = driver.read_encoder(3)
        pos4 = driver.read_encoder(4)
        print(f"Initial  addr=3: {pos3:.4f} rad  addr=4: {pos4:.4f} rad")

        targets = {
            3: pos3 - DELTA_RAD,
            4: pos4 - DELTA_RAD,
        }
        print(f"Targets  addr=3: {targets[3]:.4f} rad  addr=4: {targets[4]:.4f} rad")

        result = driver.send_joint_position_fast(targets, max_pulses=PULSES_BACK)
        print(f"Send result: {result}")

        time.sleep(0.1)

        pos3_after = driver.read_encoder(3)
        pos4_after = driver.read_encoder(4)
        print(f"After    addr=3: {pos3_after:.4f} rad  addr=4: {pos4_after:.4f} rad")

        print("\n--- ACK check (move_position with check_ack=True) ---")
        for addr in (3):
            status = driver.move_position(addr, PULSES_BACK, direction=1, check_ack=True)
            if status == 0:
                print(f"  addr={addr}: ACK received — status=FAIL (motor rejected command)")
            elif status == 1:
                print(f"  addr={addr}: ACK received — status=STARTING (motor accepted)")
            elif status == 2:
                print(f"  addr={addr}: ACK received — status=COMPLETE")
            else:
                print(f"  addr={addr}: ACK NOT received or unreadable (status={status})")


if __name__ == "__main__":
    main()
