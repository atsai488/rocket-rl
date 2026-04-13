#ifndef SERVO_H
#define SERVO_H

#include <stdint.h>

typedef struct {
    uint16_t pulse_us[12];
} CmdServoPositions;

void     servo_init(void);
void     servo_set_pulse(uint8_t idx, uint16_t pulse_us);
uint16_t servo_get_pulse(uint8_t idx);
void     servo_centre_all(void);

#endif