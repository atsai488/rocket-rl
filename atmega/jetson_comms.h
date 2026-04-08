#ifndef JETSON_COMMS_H
#define JETSON_COMMS_H

#include <stdint.h>
#include "config.h"

typedef enum {
    CMD_STEPPER_POSITIONS = 0x10,
    CMD_SERVO_POSITIONS   = 0x20,
    CMD_STEPPER_SPEED     = 0x30,
    CMD_STOP_ALL          = 0x40,
    CMD_HOME              = 0x50,
    CMD_STATUS_REQUEST    = 0x60,
} CmdType;

typedef struct {
    int32_t  step_pos[6];
    uint16_t servo_pulse[12];
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
    int16_t  mx, my, mz;
    uint8_t  flags;
} RobotStatus;

void    jetson_comms_init(void);
uint8_t jetson_comms_receive(uint8_t *cmd_type, uint8_t *payload);
void    jetson_comms_send_status(const RobotStatus *s);
void    jetson_comms_send_ack(uint8_t cmd);
void    jetson_comms_send_nack(uint8_t cmd, uint8_t err);

#endif