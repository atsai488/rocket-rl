#include <avr/io.h>
#include <avr/interrupt.h>
#include "servo.h"
#include "config.h"

static volatile uint16_t g_pulse_us[NUM_SERVOS];

void servo_init(void) {
    DDRD |= (1 << PD5) | (1 << PD6);

    ICR1   = 39999;
    OCR1A  = 3000;
    OCR1B  = 3000;
    TCCR1A = (1 << COM1A1) | (1 << COM1B1) | (1 << WGM11);
    TCCR1B = (1 << WGM13)  | (1 << WGM12)  | (1 << CS11);

    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        g_pulse_us[i] = SERVO_PULSE_MID;
    }

    /* set up GPIO outputs for remaining servo pins (direction registers + initial LOW state) */

    /* choose Timer2 prescaler + overflow/compare timing to get ~20ms servo frame or suitable ISR tick */

    /* ensure pin mapping avoids PD4 OR redesign pin assignment to prevent RS485 direction collision */

    /* create ISR to generate software PWM timing for remaining servos */
}

void servo_set_pulse(uint8_t idx, uint16_t pulse_us) {
    if (idx >= NUM_SERVOS) return;
    if (pulse_us < SERVO_PULSE_MIN) pulse_us = SERVO_PULSE_MIN;
    if (pulse_us > SERVO_PULSE_MAX) pulse_us = SERVO_PULSE_MAX;
    g_pulse_us[idx] = pulse_us;

    uint16_t ocr = pulse_us * 2;

    switch (idx) {
        case 0: OCR1A = ocr; return;
        case 1: OCR1B = ocr; return;
        default:
            /* update shared software PWM buffer/state used by ISR timing loop */
            break;
    }
}

uint16_t servo_get_pulse(uint8_t idx) {
    if (idx >= NUM_SERVOS) return SERVO_PULSE_MID;
    return g_pulse_us[idx];
}

void servo_centre_all(void) {
    for (uint8_t i = 0; i < NUM_SERVOS; i++) {
        servo_set_pulse(i, SERVO_PULSE_MID);
    }
}