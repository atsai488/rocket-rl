#include <avr/io.h>
#include <util/delay.h>
#include <limits.h>
#include "stepper.h"
#include "config.h"

static const uint8_t ALL_IDS[NUM_STEPPERS] = {
    STEPPER_ID_1, STEPPER_ID_2, STEPPER_ID_3,
    STEPPER_ID_4, STEPPER_ID_5, STEPPER_ID_6
};

static void uart1_tx(uint8_t b) {
    while (!(UCSR1A & (1 << UDRE1)));
    UDR1 = b;
}

static void uart1_flush_tx(void) {
    while (!(UCSR1A & (1 << TXC1)));
    UCSR1A |= (1 << TXC1);
}

static uint8_t uart1_rx(uint8_t *out, uint16_t timeout) {
    while (!(UCSR1A & (1 << RXC1))) {
        if (--timeout == 0) return 0;
        _delay_us(10);
    }
    *out = UDR1;
    return 1;
}

static uint8_t calc_crc(uint8_t addr, uint8_t cmd,
                         const uint8_t *data, uint8_t dlen) {
    uint8_t crc = 0xFA + addr + cmd;
    for (uint8_t i = 0; i < dlen; i++) crc += data[i];
    return crc;
}

static void send_frame(uint8_t addr, uint8_t cmd,
                        const uint8_t *data, uint8_t dlen) {
    uint8_t crc = calc_crc(addr, cmd, data, dlen);

    RS485_TX();
    _delay_us(4);

    uart1_tx(0xFA);
    uart1_tx(addr);
    uart1_tx(cmd);
    for (uint8_t i = 0; i < dlen; i++) uart1_tx(data[i]);
    uart1_tx(crc);

    uart1_flush_tx();
    _delay_us(4);
    RS485_RX();
}

#define RX_TIMEOUT 8000

static uint8_t recv_response(uint8_t *buf, uint8_t total_len) {
    uint8_t b;
    if (!uart1_rx(&b, RX_TIMEOUT)) return 0;
    if (b != 0xFB) return 0;
    buf[0] = b;
    for (uint8_t i = 1; i < total_len; i++) {
        if (!uart1_rx(&buf[i], RX_TIMEOUT)) return 0;
    }
    return total_len;
}

void stepper_init(uint16_t ubrr) {
    RS485_DDR  |= (1 << RS485_DIR_PIN);
    RS485_RX();

    UBRR1H = (uint8_t)(ubrr >> 8);
    UBRR1L = (uint8_t)(ubrr);
    UCSR1B = (1 << TXEN1) | (1 << RXEN1);
    UCSR1C = (3 << UCSZ10);
}

void stepper_enable(uint8_t id, uint8_t on) {
    uint8_t d[1] = { on ? 0x01 : 0x00 };
    send_frame(id, 0xF3, d, 1);

    uint8_t buf[4];
    recv_response(buf, 4);
    _delay_ms(5);
}

void stepper_move_to_pulses(uint8_t id, uint8_t dir,
                             uint16_t speed, uint8_t acc,
                             uint32_t pulses) {
    if (speed > 1600) speed = 1600;
    if (acc   > 32)   acc   = 32;

    uint8_t byte4 = ((dir & 0x01) << 7) | ((speed >> 8) & 0x0F);
    uint8_t byte5 = (uint8_t)(speed & 0xFF);

    uint8_t d[7];
    d[0] = byte4;
    d[1] = byte5;
    d[2] = acc;
    d[3] = (uint8_t)(pulses >> 24);
    d[4] = (uint8_t)(pulses >> 16);
    d[5] = (uint8_t)(pulses >>  8);
    d[6] = (uint8_t)(pulses);

    send_frame(id, 0xFD, d, 7);

    uint8_t buf[4];
    recv_response(buf, 4);
}

#define MICROSTEPS_PER_REV 3200UL

void stepper_move_degrees(uint8_t id, uint8_t dir,
                           uint16_t speed, uint8_t acc,
                           float degrees) {
    uint32_t pulses = (uint32_t)((degrees / 360.0f) * MICROSTEPS_PER_REV);
    stepper_move_to_pulses(id, dir, speed, acc, pulses);
}

void stepper_stop(uint8_t id, uint8_t acc) {
    uint8_t d[7] = { 0x00, 0x00, acc, 0x00, 0x00, 0x00, 0x00 };
    send_frame(id, 0xFD, d, 7);

    uint8_t buf[4];
    recv_response(buf, 4);
}

void stepper_stop_all(void) {
    for (uint8_t i = 0; i < NUM_STEPPERS; i++) {
        stepper_stop(ALL_IDS[i], 0);
        _delay_us(500);
    }
}

int32_t stepper_get_position(uint8_t id) {
    send_frame(id, 0x30, NULL, 0);

    uint8_t buf[10];
    if (!recv_response(buf, 10)) return INT32_MIN;

    int32_t carry = ((int32_t)buf[3] << 24)
                  | ((int32_t)buf[4] << 16)
                  | ((int32_t)buf[5] <<  8)
                  |  (int32_t)buf[6];

    uint16_t value = ((uint16_t)buf[7] << 8) | buf[8];

    return carry * (int32_t)0x4000 + (int32_t)value;
}

void stepper_set_zero(uint8_t id) {
    /* add implementation */
    (void)id;
}

void stepper_calibrate(uint8_t id) {
    uint8_t d[1] = { 0x00 };
    send_frame(id, 0x80, d, 1);

    uint8_t buf[4];
    uint8_t status = 0;
    do {
        _delay_ms(200);
        recv_response(buf, 4);
        status = buf[3];
    } while (status == 0);
}