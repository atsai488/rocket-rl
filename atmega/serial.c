#include <avr/io.h>
#include "serial.h"

void serial_init(unsigned short ubrr) {
    UBRR0H = (unsigned char)(ubrr >> 8);
    UBRR0L = (unsigned char)(ubrr);
    UCSR0B = (1 << TXEN0) | (1 << RXEN0);
    UCSR0C = (3 << UCSZ00);
}

static void serial_out(char c) {
    while (!(UCSR0A & (1 << UDRE0)));
    UDR0 = c;
}

void serial_out_string(const char *str) {
    while (*str) serial_out(*str++);
    serial_out('\r');
    serial_out('\n');
}