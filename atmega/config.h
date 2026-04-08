#ifndef CONFIG_H
#define CONFIG_H

#define F_CPU 16000000UL

#define UBRR_JETSON   8
#define UBRR_STEPPER  25

#define RS485_DDR     DDRD
#define RS485_PORT    PORTD
#define RS485_DIR_PIN PD4
#define RS485_TX()    (RS485_PORT |=  (1 << RS485_DIR_PIN))
#define RS485_RX()    (RS485_PORT &= ~(1 << RS485_DIR_PIN))

#define I2C_BDIV  72

#define MKS_HEAD   0xFA
#define MKS_TAIL   0xFB

#define MKS_CMD_ENABLE    0xF3
#define MKS_CMD_SET_POS   0xFD
#define MKS_CMD_SET_SPEED 0xF6
#define MKS_CMD_ESTOP     0xF7
#define MKS_CMD_READ_ENC  0x31
#define MKS_CMD_SET_ZERO  0x92

#define STEPPER_ID_1   0x01
#define STEPPER_ID_2   0x02
#define STEPPER_ID_3   0x03
#define STEPPER_ID_4   0x04
#define STEPPER_ID_5   0x05
#define STEPPER_ID_6   0x06

#define NUM_STEPPERS   6

#define STEPPER_DEFAULT_SPEED  200
#define STEPPER_DEFAULT_ACCEL  50

#define NUM_SERVOS        12
#define SERVO_PULSE_MIN   1000
#define SERVO_PULSE_MID   1500
#define SERVO_PULSE_MAX   2000

#define PKT_START  0xAA
#define PKT_END    0x55
#define PKT_MAX_DATA  64

#define CMD_STEPPER_POSITIONS  0x10
#define CMD_SERVO_POSITIONS    0x20
#define CMD_STEPPER_SPEED      0x30
#define CMD_STOP_ALL           0x40
#define CMD_HOME               0x50
#define CMD_STATUS_REQUEST     0x60

#define RESP_STATUS   0x80
#define RESP_ACK      0x81
#define RESP_NACK     0x82

#define CMD_PAYLOAD_LEN  64

#define FLAG_STEPPERS_MOVING  (1 << 0)
#define FLAG_ESTOP            (1 << 1)
#define FLAG_STEPPER_ERROR    (1 << 2)
#define FLAG_IMU_ERROR        (1 << 3)

#endif