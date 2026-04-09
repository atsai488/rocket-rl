#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/delay.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#include "config.h"
#include "serial.h"
#include "i2c.h"
#include "imu.h"
#include "stepper.h"
#include "servo.h"
#include "jetson_comms.h"

#define STATUS_INTERVAL_MS 20

static volatile uint32_t ms_ticks = 0;

ISR(TIMER0_COMPA_vect) {
    ms_ticks++;
}

static void timer0_init(void) {
    TCCR0A = (1 << WGM01);
    TCCR0B = (1 << CS01) | (1 << CS00);
    OCR0A  = 249;
    TIMSK0 = (1 << OCIE0A);
}

static uint32_t get_ms(void) {
    uint32_t t;
    cli();
    t = ms_ticks;
    sei();
    return t;
}

static RobotStatus g_status;
static uint8_t     g_flags  = 0;
static uint8_t     g_estop  = 0;

static int32_t  g_target_pos[NUM_STEPPERS] = {0};
static uint16_t g_target_speed             = STEPPER_DEFAULT_SPEED;
static uint8_t  g_target_accel             = STEPPER_DEFAULT_ACCEL;

static const uint8_t STEPPER_IDS[NUM_STEPPERS] = {
    STEPPER_ID_RF, STEPPER_ID_RM, STEPPER_ID_RB,
    STEPPER_ID_LF, STEPPER_ID_LM, STEPPER_ID_LB,
};

static void handle_stepper_positions(const uint8_t *payload) {
    const CmdStepperPositions *cmd = (const CmdStepperPositions *)payload;
    if (g_estop) return;
    for (int i = 0; i < NUM_STEPPERS; i++) {
        g_target_pos[i] = cmd->pos[i];
        stepper_move_to(STEPPER_IDS[i], cmd->pos[i], cmd->speed, cmd->accel);
    }
    g_target_speed = cmd->speed;
    g_target_accel = cmd->accel;
    g_flags |= FLAG_STEPPERS_MOVING;
}

static void handle_servo_positions(const uint8_t *payload) {
    const CmdServoPositions *cmd = (const CmdServoPositions *)payload;
    for (int i = 0; i < NUM_SERVOS; i++) {
        servo_set_pulse(i, cmd->pulse_us[i]);
    }
}

static void handle_stepper_speed(const uint8_t *payload) {
    const CmdStepperSpeed *cmd = (const CmdStepperSpeed *)payload;
    for (int i = 0; i < NUM_STEPPERS; i++) {
        g_target_speed = cmd->speed[i];
        g_target_accel = cmd->accel[i];
    }
}

static void handle_stop_all(void) {
    stepper_stop_all();
    g_estop  = 1;
    g_flags |= FLAG_ESTOP;
    g_flags &= ~FLAG_STEPPERS_MOVING;
    serial_out_string("ESTOP\r\n");
}

static void handle_home(void) {
    g_estop = 0;
    g_flags &= ~FLAG_ESTOP;
    for (int i = 0; i < NUM_STEPPERS; i++) {
        stepper_set_zero(STEPPER_IDS[i]);
        g_target_pos[i] = 0;
    }
    servo_centre_all();
    serial_out_string("HOMED\r\n");
}

static void build_and_send_status(void) {
    for (int i = 0; i < NUM_STEPPERS; i++) {
        int32_t pos = stepper_get_position(STEPPER_IDS[i]);
        if (pos == INT32_MIN) {
            g_flags |= FLAG_STEPPER_ERROR;
            pos = g_target_pos[i];
        }
        g_status.step_pos[i] = pos;
    }

    IMUData imu_data;
    if (imu_read(&imu_data) != 0) {
        g_flags |= FLAG_IMU_ERROR;
    } else {
        g_status.ax = imu_data.ax;  g_status.ay = imu_data.ay;  g_status.az = imu_data.az;
        g_status.gx = imu_data.gx;  g_status.gy = imu_data.gy;  g_status.gz = imu_data.gz;
        g_status.mx = imu_data.mx;  g_status.my = imu_data.my;  g_status.mz = imu_data.mz;
    }

    for (int i = 0; i < NUM_SERVOS; i++) {
        g_status.servo_pulse[i] = servo_get_pulse(i);
    }

    g_status.flags = g_flags;
    jetson_comms_send_status(&g_status);
}

int main(void) {
    serial_init(UBRR_JETSON);
    i2c_init(BDIV);
    stepper_init(UBRR_STEPPER);
    servo_init();
    timer0_init();
    jetson_comms_init();

    _delay_ms(200);

    if (imu_init() != 0) {
        serial_out_string("IMU INIT FAIL\r\n");
        g_flags |= FLAG_IMU_ERROR;
    }

    for (int i = 0; i < NUM_STEPPERS; i++) {
        stepper_enable(STEPPER_IDS[i], 1);
        _delay_ms(10);
    }

    for (int i = 0; i < NUM_STEPPERS; i++) {
        stepper_set_zero(STEPPER_IDS[i]);
    }

    servo_centre_all();

    sei();

    serial_out_string("ROBOT ONLINE\r\n");

    uint32_t last_status_ms = 0;
    uint8_t  cmd_type       = 0;
    uint8_t  cmd_payload[CMD_PAYLOAD_LEN];

    while (1) {
        if (jetson_comms_receive(&cmd_type, cmd_payload)) {
            switch ((CmdType)cmd_type) {
                case CMD_COMBINED: {
                    const CmdCombined *cmd = (const CmdCombined *)cmd_payload;
                    if (!g_estop) {
                        for (int i = 0; i < 4; i++) {
                            float steps = cmd->stepper[i];
                            uint8_t  dir    = (steps >= 0.0f) ? 1 : 0;
                            uint32_t pulses = (uint32_t)(steps < 0.0f ? -steps : steps);
                            stepper_move_to_pulses(STEPPER_IDS[i], dir,
                                                   g_target_speed, g_target_accel, pulses);
                        }
                    }
                    for (int i = 0; i < 2; i++) {
                        uint16_t pulse = (uint16_t)(SERVO_PULSE_MIN
                            + (uint16_t)((cmd->servo[i] / 180.0f)
                                         * (SERVO_PULSE_MAX - SERVO_PULSE_MIN)));
                        servo_set_pulse(i, pulse);
                    }
                    break;
                }
                case CMD_STOP_ALL:           handle_stop_all();                     break;
                case CMD_HOME:               handle_home();                         break;
                case CMD_STATUS_REQUEST:
                    build_and_send_status();
                    last_status_ms = get_ms();
                    break;
                default:
                    serial_out_string("UNKNOWN CMD\r\n");
                    break;
            }
        }

        uint32_t now = get_ms();
        if ((now - last_status_ms) >= STATUS_INTERVAL_MS) {
            last_status_ms = now;
            build_and_send_status();
        }
    }

    return 0;
}

