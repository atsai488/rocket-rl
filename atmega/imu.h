#ifndef IMU_H
#define IMU_H

#include <stdint.h>

typedef struct {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    int16_t mx, my, mz;
} IMUData;

uint8_t imu_init(void);
uint8_t imu_read(IMUData *data);

#endif