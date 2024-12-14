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

| Opcode | Instruction | Description |
|--------|-------------|-------------|
| 0x00   | NOP         | No operation |
| 0x01   | ADD         | Add two registers |
| 0x02   | SUB         | Subtract two registers |
| 0x03   | MUL         | Multiply two registers |
| 0x04   | DIV         | Divide two registers |
| 0x05   | AND         | AND between two registers (bitwise) |
| 0x06   | OR          | OR between two registers (bitwise) |
| 0x07   | XOR         | XOR between two registers (bitwise) |
| 0x08   | NOT         | NOT between two registers (bitwise) |
| 0x09   | SHL         | Shift left (*2) |
| 0x0A   | SHR         | Shift right (/2) |
| 0x0B   | LOAD        | Load data from memory to a register |
| 0x0C   | STORE       | Store data from a register to memory |
| 0x0D   | MOV         | Move data between registers |
| 0x0E   | JMP         | Jump to an address |
| 0x0F   | CALL        | Call a function or subroutine |
| 0x10   | RET         | Return from function or subroutine |
| 0x11   | JE          | Jump if equal |
| 0x12   | JNE         | Jump if not equal |
| 0x13   | JG          | Jump if greater than |
| 0x14   | JL          | Jump if less than |
| 0x15   | JGE         | Jump if greater or equal |
| 0x16   | JLE         | Jump if less or equal |
| 0x17   | IN          | Read data from an I/O port (Not in use) |
| 0x18   | OUT         | Write data to an I/O port (Not in use) |
| 0x19   | PUSH        | Push data onto the stack |
| 0x1A   | POP         | Pop data from the stack |
| 0x1B   | INT         | Trigger a software interrupt |
| 0x1C   | HALT        | Halt the execution |
| 0x1D   | IRET        | Returns from an interrupt |
| 0x1E   | RES2        | Reserved |
| 0x1F   | RES3        | Reserved |