#ifndef STEPPER_H
#define STEPPER_H

#include <stdint.h>
#include <limits.h>

typedef struct {
    int32_t  pos_degrees[6];
    uint16_t speed[6];
    uint8_t  acc[6];
    uint8_t  dir[6];
} CmdStepperPositions;

typedef struct {
    uint16_t speed[6];
    uint8_t  acc[6];
} CmdStepperSpeed;

void stepper_init(uint16_t ubrr);

void stepper_enable(uint8_t motor_id, uint8_t on);

void stepper_move_to_pulses(uint8_t motor_id, uint8_t dir,
                             uint16_t speed, uint8_t acc,
                             uint32_t pulses);

void stepper_move_degrees(uint8_t motor_id, uint8_t dir,
                           uint16_t speed, uint8_t acc,
                           float degrees);

void stepper_stop(uint8_t motor_id, uint8_t acc);

void stepper_stop_all(void);

int32_t stepper_get_position(uint8_t motor_id);

void stepper_set_zero(uint8_t motor_id);

void stepper_calibrate(uint8_t motor_id);

#endif