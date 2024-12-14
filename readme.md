# SIMPLE-VM

SIMPLE-VM is a VM based on the custom SIMPLE architecture.

## Use

1. Download the executable from the releases path.
2. Place the executable in your PATH
3. Run ```simple {path/to/file/containing/machine/code}``` in your terminal

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
| 00   | NOP         | No operation |
| 01   | ADD         | Add two registers |
| 02   | SUB         | Subtract two registers |
| 03   | MUL         | Multiply two registers |
| 04   | DIV         | Divide two registers |
| 05   | AND         | AND between two registers (bitwise) |
| 06   | OR          | OR between two registers (bitwise) |
| 07   | XOR         | XOR between two registers (bitwise) |
| 08   | NOT         | NOT between two registers (bitwise) |
| 09   | SHL         | Shift left (*2) |
| 0A   | SHR         | Shift right (/2) |
| 0B   | LOAD        | Load data from memory to a register |
| 0C   | STORE       | Store data from a register to memory |
| 0D   | MOV         | Move data between registers |
| 0E   | JMP         | Jump to an address |
| 0F   | CALL        | Call a function or subroutine |
| 10   | RET         | Return from function or subroutine |
| 11   | JE          | Jump if equal |
| 12   | JNE         | Jump if not equal |
| 13   | JG          | Jump if greater than |
| 14   | JL          | Jump if less than |
| 15   | JGE         | Jump if greater or equal |
| 16   | JLE         | Jump if less or equal |
| 17   | IN          | Read data from an I/O port (Not in use) |
| 18   | OUT         | Write data to an I/O port (Not in use) |
| 19   | PUSH        | Push data onto the stack |
| 1A   | POP         | Pop data from the stack |
| 1B   | INT         | Trigger a software interrupt |
| 1C   | HALT        | Halt the execution |
| 1D   | IRET        | Returns from an interrupt |
| 1E   | RES2        | Reserved |
| 1F   | RES3        | Reserved |