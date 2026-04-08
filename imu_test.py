import time
import board
import busio
import adafruit_bno055

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

while True:
    print("Euler:", sensor.euler)
    print("Gyro:", sensor.gyro)
    print("Accel:", sensor.acceleration)
    print("Mag:", sensor.magnetic)
    print("------------------")
    time.sleep(1)
