# Simple Instruction Machine for Programming and Logic Engineering - Virtual Machine

SIMPLE-VM is a VM based on the custom SIMPLE architecture.
SIMPLE is a RISC 32-bit microcontroller architecture.

## Use

1. Download the executable from the releases path.
2. Place the executable in your PATH
3. Run ```simple {path/to/your/file/containing/simple-asm}``` in your terminal

### Debug Modes

You can append the following debug modes to your terminal command, this will print their states at the beginning and the end of the program.

- ```--registers``` or ```-r```: prints the state of the general purpose and special registers\n
- ```--stack``` or ```-s```: prints the state of the stack
- ```--program_memory``` or ```-p```: prints the state of program memory
- ```--data_memory``` or ```-d```: prints the state of data memory

## Memory Layout
- Program Memory: 32 KB
- Data Memory: 8 KB
- Stack: 1 KB

## Registers
- General Purpose: R0-R7 (32-bit)
- Special Purpose:
  - PC (Program Counter)
  - SP (Stack Pointer)
  - SR (Status Register/FLAGS)

## Instruction Format
32-bit instruction word:
```
[8-bit opcode][4-bit src][4-bit dest][16-bit immediate]
```

15 inside the 4-bit src (F in hexidecimal) indicates that the immediate value is used instead of a src register.

## Status Flags
- Zero (Z): Set when result is zero
- Negative (N): Set when result is negative
- Carry (C): Set on arithmetic overflow/underflow

## Current Instruction Set

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

## simple-asm

### Instruction Format
Each instruction follows this general format:
```
OPCODE [operand1] [operand2]
```

### Operand Types
- **Registers**: `R0` through `R7`
- **Immediate Values**: 
  - Decimal: `#123` 
  - Hexadecimal: `#0xFF`
  - Binary: `#0b1010`
- **Labels**: 
  - Definition: `label_name:`
  - Reference: `@label_name`

### Instruction Set

#### Data Movement
- `MOV src dest` - Move data between registers or load immediate value
  - Example: `MOV #42 R0` (Load 42 into R0)
  - Example: `MOV R0 R1` (Copy R0 to R1)

#### Arithmetic Operations
- `ADD src dest` - Add src to dest, store in dest
- `SUB src dest` - Subtract src from dest, store in dest
- `MUL src dest` - Multiply dest by src, store in dest
- `DIV src dest` - Divide dest by src, store in dest

#### Logical Operations
- `AND src dest` - Bitwise AND
- `OR src dest` - Bitwise OR
- `XOR src dest` - Bitwise XOR
- `NOT src dest` - Bitwise NOT
- `SHL src dest` - Shift left
- `SHR src dest` - Shift right

#### Memory Operations
- `LOAD src dest` - Load from memory address in src to dest register
- `STORE src dest` - Store value from src register to memory address in dest
- `PUSH src` - Push register or immediate value onto stack
- `POP dest` - Pop value from stack into register

#### Control Flow
- `JMP addr` - Unconditional jump
- `CALL addr` - Call subroutine
- `RET` - Return from subroutine
- `JE addr` - Jump if equal (zero flag set)
- `JNE addr` - Jump if not equal (zero flag clear)
- `JG addr` - Jump if greater
- `JL addr` - Jump if less
- `JGE addr` - Jump if greater or equal
- `JLE addr` - Jump if less or equal

#### System Operations
- `NOP` - No operation
- `HALT` - Stop execution
- `INT vector` - Trigger software interrupt (vector: 0-255)
- `IRET` - Return from interrupt
- `IN` - Input operation (reserved for future use)
- `OUT` - Output operation (reserved for future use)

### Comments
Comments start with semicolon (;) and continue to the end of the line:
```
MOV #42 R0  ; Load 42 into R0
```

### Labels
Labels can be used to mark positions in code for jumps and calls:
```
loop:           ; Define label
    ADD #1 R0   ; Increment R0
    JMP @loop   ; Jump back to loop
```

### Example Program
```
; Calculate sum of 3 and 4
    MOV #3 R0       ; Load 3 into R0
    MOV #4 R1       ; Load 4 into R1
    ADD R0 R1       ; Add R0 to R1, result in R1
    HALT            ; Stop execution
```

### Error Handling
The assembler will report errors for:
- Invalid opcodes
- Invalid register numbers
- Invalid immediate values
- Undefined labels
- Syntax errors
- Memory access out of bounds