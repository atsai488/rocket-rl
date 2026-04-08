#include <avr/io.h>
#include <util/delay.h>
#include "imu.h"
#include "i2c.h"

/* setup implementation blank rn*/ 

uint8_t imu_init(void) {
    return 0;
}

uint8_t imu_read(IMUData *data) {
    (void)data;
    return 0;
}