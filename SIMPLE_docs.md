# SIMPLE Architecture Documentation
## Simple Instruction Machine for Programming and Logic Engineering

SIMPLE is a 32-bit microcontroller architecture.

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
[8-bit opcode][4-bit dest][4-bit src][16-bit immediate]
```

### Status Flags
- Zero (Z): Set when result is zero
- Negative (N): Set when result is negative
- Carry (C): Set on arithmetic overflow/underflow

### Current Instruction Set

| Opcode (Hex) | Mnemonic | Description | Format |
|--------------|----------|-------------|---------|
| 0x00 | NOP | No operation | NOP |
| 0x01 | ADD | Add registers | ADD Rd, Rs |
| 0x02 | SUB | Subtract registers | SUB Rd, Rs |
| 0x03 | MUL | Multiply registers | MUL Rd, Rs |
| 0x04 | DIV | Divide registers | DIV Rd, Rs |
| 0x05 | AND | Bitwise AND | AND Rd, Rs |
| 0x06 | OR | Bitwise OR | OR Rd, Rs |
| 0x07 | XOR | Bitwise XOR | XOR Rd, Rs |
| 0x08 | NOT | Bitwise NOT | NOT Rd, Rs |
| 0x09 | SHL | Shift left | SHL Rd, Rs |
| 0x0A | SHR | Shift right | SHR Rd, Rs |
| 0x0B | LOAD | Load from memory | LOAD Rd, [Rs+imm] |
| 0x0C | STORE | Store to memory | STORE Rs, [Rd+imm] |
| 0x0D | MOV | Move/load immediate | MOV Rd, #imm or MOV Rd, Rs |
| 0x1C | HALT | Halt execution | HALT |
