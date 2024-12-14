# SIMPLE-VM

SIMPLE-VM is a VM based on the custom SIMPLE architecture.

## Simple Instruction Machine for Programming and Logic Engineering

SIMPLE is a RISC 32-bit microcontroller architecture.

### Memory Layout
- Program Memory: 32 KB
- Data Memory: 8 KB
- Stack: 1 KB
- I/O Ports: 8 ports (mapped to highest addresses of data memory)

### Registers
- General Purpose: R0-R7 (32-bit)
- Special Purpose:
  - PC (Program Counter)
  - SP (Stack Pointer)
  - SR (Status Register/FLAGS)
  - PP (Peripheral Pointer)

### Instruction Format
32-bit instruction word:
```
[8-bit opcode][4-bit src][4-bit dest][16-bit immediate]
```

15 inside the 4-bit src (F in hexidecimal) indicates that the immediate value is used instead of a src register.

### Status Flags
- Zero (Z): Set when result is zero
- Negative (N): Set when result is negative
- Carry (C): Set on arithmetic overflow/underflow

### Current Instruction Set

TODO