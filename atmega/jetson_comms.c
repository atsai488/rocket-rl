#include <avr/io.h>
#include <string.h>
#include "jetson_comms.h"
#include "config.h"

void jetson_comms_init(void) {
    UBRR0H = (uint8_t)(UBRR_JETSON >> 8);
    UBRR0L = (uint8_t)(UBRR_JETSON);
    UCSR0B = (1 << TXEN0) | (1 << RXEN0);
    UCSR0C = (3 << UCSZ00);
}

static void tx(uint8_t b) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = b;
}

static uint8_t rx(uint8_t *b, uint16_t timeout) {
    while (!(UCSR0A & (1 << RXC0))) {
        if (--timeout == 0) return 0;
    }
    *b = UDR0;
    return 1;
}

static void send_packet(uint8_t cmd, const uint8_t *data, uint8_t len) {
    uint8_t csum = cmd ^ len;
    for (uint8_t i = 0; i < len; i++) csum ^= data[i];

    tx(PKT_START);
    tx(cmd);
    tx(len);
    for (uint8_t i = 0; i < len; i++) tx(data[i]);
    tx(csum);
    tx(PKT_END);
}

void jetson_comms_send_ack(uint8_t cmd) {
    send_packet(RESP_ACK, &cmd, 1);
}

void jetson_comms_send_nack(uint8_t cmd, uint8_t err) {
    uint8_t d[2] = { cmd, err };
    send_packet(RESP_NACK, d, 2);
}

void jetson_comms_send_status(const RobotStatus *s) {
    uint8_t buf[67];
    uint8_t idx = 0;

    for (uint8_t i = 0; i < 6; i++) {
        buf[idx++] = (uint8_t)((uint32_t)s->step_pos[i] >> 24);
        buf[idx++] = (uint8_t)((uint32_t)s->step_pos[i] >> 16);
        buf[idx++] = (uint8_t)((uint32_t)s->step_pos[i] >>  8);
        buf[idx++] = (uint8_t)((uint32_t)s->step_pos[i]);
    }

    for (uint8_t i = 0; i < 12; i++) {
        buf[idx++] = (uint8_t)(s->servo_pulse[i] >> 8);
        buf[idx++] = (uint8_t)(s->servo_pulse[i]);
    }

    int16_t imu_vals[9] = {
        s->ax, s->ay, s->az,
        s->gx, s->gy, s->gz,
        s->mx, s->my, s->mz
    };

    for (uint8_t i = 0; i < 9; i++) {
        buf[idx++] = (uint8_t)((uint16_t)imu_vals[i] >> 8);
        buf[idx++] = (uint8_t)((uint16_t)imu_vals[i]);
    }

    buf[idx++] = s->flags;

    send_packet(RESP_STATUS, buf, idx);
}

uint8_t jetson_comms_receive(uint8_t *cmd_type, uint8_t *payload) {
    uint8_t b, len;

    if (!rx(&b, 1)) return 0;
    if (b != PKT_START) return 0;

    if (!rx(cmd_type, 5000)) return 0;
    if (!rx(&len, 5000)) return 0;
    if (len > PKT_MAX_DATA) return 0;

    uint8_t csum = *cmd_type ^ len;
    for (uint8_t i = 0; i < len; i++) {
        if (!rx(&payload[i], 5000)) return 0;
        csum ^= payload[i];
    }

    uint8_t rx_csum, end;
    if (!rx(&rx_csum, 5000)) return 0;
    if (!rx(&end, 5000)) return 0;

    if (rx_csum != csum) return 0;
    if (end != PKT_END) return 0;

    return 1;
}