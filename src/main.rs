const PROGRAM_MEMORY_SIZE: usize = 1024 * 32; // 32 KB
const DATA_MEMORY_SIZE: usize = 1024 * 8;     // 8 KB
const STACK_SIZE: usize = 1024;           // 256 bytes

/*
    NOP = 0x00,   // No operation
    ADD = 0x01,   // Add two registers
    SUB = 0x02,   // Subtract two registers
    MUL = 0x03,   // Multiply two registers
    DIV = 0x04,   // Divide two registers
    AND = 0x05,   // AND between two registers (bitwise)
    OR = 0x06,    // OR between two registers (bitwise)
    XOR = 0x07,   // XOR between two registers (bitwise)
    NOT = 0x08,   // NOT between two registers (bitwise)
    SHL = 0x09,   // Shift left (*2)
    SHR = 0x0A,   // Shift right (/2)
    LOAD = 0x0B,  // Load data from memory to a register
    STORE = 0x0C, // Store data from a register to memory
    MOV = 0x0D,   // Move data between registers
    JMP = 0x0E,   // Jump to an address
    CALL = 0xF,   // Call a function or subroutine
    RET = 0x10,   // Return from function or subroutine
    JE = 0x11,    // Jump if equal
    JNE	= 0x12,   // Jump if not equal
    JG = 0x13,    // Jump if greater than
    JL	= 0x14,   // Jump if less than
    JGE	= 0x15,   // Jump if greater or equal
    JLE	= 0x16,   // Jump if less or equal
    IN	= 0x17,   // Read data from an I/O port
    OUT	= 0x18,   // Write data to an I/O port
    PUSH = 0x19,  // Push data onto the stack
    POP	= 0x1A,   // Pop data from the stack
    INT	= 0x1B,   // Trigger a software interrupt
    HALT = 0x1C,  // Halt the execution
    IRET = 0x1D,  // Returns from an interrupt
    RES2 = 0x1E,  // Reserved
    RES3 = 0x1F   // Reserved
*/

#[derive(Debug)]
struct Instruction {
    opcode: u8,           // 8 bits for the opcode
    src_reg: u8,          // 4 bits for the source register
    dest_reg: u8,         // 4 bits for the destination register
    immediate: u16,       // 16 bits for immediate value
}

impl Instruction {
    fn from_opcode(opcode: u32) -> Self {
        let opcode_value = (opcode >> 24) as u8;              // Extract the 8-bit opcode
        let src_reg = ((opcode >> 20) & 0x0F) as u8;          // Extract the 4-bit source register
        let dest_reg = ((opcode >> 16) & 0x0F) as u8;         // Extract the 4-bit destination register
        let immediate = (opcode & 0xFFFF) as u16;            // Extract the 16-bit immediate value
    
        Instruction {
            opcode: opcode_value,
            src_reg,
            dest_reg,
            immediate,
        }
    }
    
}

#[derive(Debug, Clone, Copy)]
struct Flags {
    zero: bool,
    negative: bool,
    carry: bool,
}

#[derive(Debug)]
struct CPU {
    running: bool,

    // Registers
    r0: u32,
    r1: u32,
    r2: u32,
    r3: u32,
    r4: u32,
    r5: u32,
    r6: u32,
    r7: u32,
    pc: u32,    // Program Counter
    sp: u32,    // Stack Pointer
    sr: Flags,  // Status Register/FLAGS
    
    // Memory
    program_memory: [u16; PROGRAM_MEMORY_SIZE],
    data_memory: [u8; DATA_MEMORY_SIZE],
    stack: [u8; STACK_SIZE],
    
    // Interrupt state
    call_stack: Vec<u32>,
    interrupt_vector: [u32; 256],
    interrupt_enabled: bool,
    interrupt_queue: Vec<u8>,
    saved_state: Option<CpuState>,    
}

#[derive(Debug, Clone)]
struct CpuState {
    registers: [u32; 8],
    pc: u32,
    sr: Flags,
}

impl CPU {
    fn new() -> Self {
        Self {
            running: true,

            r0: 0,
            r1: 0,
            r2: 0,
            r3: 0,
            r4: 0,
            r5: 0,
            r6: 0,
            r7: 0,
            pc: 0,
            sp: STACK_SIZE as u32, 
            sr: Flags {
                zero: false,
                carry: false,
                negative: false,
            },

            program_memory: [0; PROGRAM_MEMORY_SIZE],
            data_memory: [0; DATA_MEMORY_SIZE],
            stack: [0; STACK_SIZE],

            call_stack: Vec::new(),
            interrupt_vector: [0; 256],
            interrupt_enabled: false,
            interrupt_queue: Vec::new(),
            saved_state: None,
        }
    }

    fn load_program(&mut self, program: &[u32]) {
        for (i, &instruction) in program.iter().enumerate() {
            let addr = i * 4;
            if addr < PROGRAM_MEMORY_SIZE {
                self.program_memory[addr / 2] = ((instruction >> 16) & 0xFFFF) as u16;
            self.program_memory[(addr / 2) + 1] = (instruction & 0xFFFF) as u16;
            }
        }
    }

    fn run(&mut self) {
        while self.running {
            self.handle_interrupts();
            let addr = (self.pc as usize) / 2;
            if addr >= PROGRAM_MEMORY_SIZE - 3 {
                eprintln!("Program counter out of bounds at {:#X}", self.pc);
                self.running = false;
                return;
            }
            
            let high = self.program_memory[addr] as u32;
            let low = self.program_memory[addr + 1] as u32;
            let instruction = (high << 16) | low;
            
            self.decode_execute(instruction);
        }
    }
    

    fn debug_registers(&self) {
        println!("R0: {:#X}, R1: {:#X}, R2: {:#X}, R3: {:#X}, R4: {:#X}, R5: {:#X}, R6: {:#X}, R7: {:#X}"
        , self.r0, self.r1, self.r2, self.r3, self.r4, self.r5, self.r6, self.r7);
        println!("PC: {:#X}, SP: {:#X}"
        , self.pc, self.sp);
        println!("Flags - Zero: {}, Negative: {}, Carry: {}"
        , self.sr.zero, self.sr.negative, self.sr.carry);
    }
    
    fn debug_stack(&self) {
        println!("sp: {}
        stack: {:?}", self.sp, self.stack);
    }

    fn debug_program_memory(&self) {
        println!("Program memory: {:?}", self.program_memory);
    }

    fn debug_data_memory(&self) {
        println!("Data memory: {:?}", self.data_memory);
    }

    fn decode_execute(&mut self, opcode: u32) {
        let instruction = Instruction::from_opcode(opcode);

        match instruction.opcode {
            0x00 => { // NOP
                // Do nothing
                self.pc += 2;
            }
            0x01 => { // ADD
                self.execute_add(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            },
            0x02 => { // SUB
                self.execute_sub(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            },
            0x03 => { // MUL
                self.execute_mul(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x04 => { // DIV
                self.execute_div(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x05 => { // AND
                self.execute_and(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x06 => { // OR
                self.execute_or(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }   
            0x07 => { // XOR
                self.execute_xor(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x08 => { // NOT
                self.execute_not(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x09 => { // SHL
                self.execute_shl(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x0A => { // SHR
                self.execute_shr(instruction.src_reg, instruction.dest_reg);
                self.pc += 2;
            }
            0x0B => { // LOAD
                self.execute_load(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 2;
            }
            0x0C => { // STORE
                self.execute_store(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 2;
            },
            0x0D => { // MOV
                self.execute_mov(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 2;
            },
            0x0E => { // JMP
                self.execute_jmp(instruction.immediate);
            },
            0x0F => { // CALL
                self.execute_call(instruction.immediate);
            },
            0x10 => { // RET
                self.execute_ret();
            },
            0x11 => { // JE
                self.execute_je(instruction.immediate);
            },
            0x12 => { // JNE
                self.execute_jne(instruction.immediate);
            },
            0x13 => { // JG
                self.execute_jg(instruction.immediate);
            },
            0x14 => { // JL
                self.execute_jl(instruction.immediate);
            },

            //..

            0x19 => { // PUSH
                self.execute_push(instruction.src_reg, instruction.immediate);
                self.pc += 2;
            },
            0x1A => { // POP
                self.execute_pop(instruction.dest_reg);
                self.pc += 2;
            },
            0x1B => { // INT
                self.execute_int(instruction.immediate as u8);
            },
            0x1C => { // HALT
                self.running = false;
            },
            0x1D => { // IRET
                self.execute_iret();
            }
            _ => {
                eprintln!("Invalid Opcode: {:#X} at PC: {:#X}. Halting.", instruction.opcode, self.pc);
                self.running = false;
            }
        }
    }

    fn read_register(&mut self, reg: u8) -> u32 {
        match reg {
            0 => self.r0,
            1 => self.r1,
            2 => self.r2,
            3 => self.r3,
            4 => self.r4,
            5 => self.r5,
            6 => self.r6,
            7 => self.r7,
            _ => { 
                eprintln!("Invalid register at {:#X}", self.pc);
                self.running = false;
                0xF
            },
        }
    }

    fn write_register(&mut self, reg: u8, value: u32) {
        match reg {
            0 => self.r0 = value,
            1 => self.r1 = value,
            2 => self.r2 = value,
            3 => self.r3 = value,
            4 => self.r4 = value,
            5 => self.r5 = value,
            6 => self.r6 = value,
            7 => self.r7 = value,
            _ => { 
                eprintln!("Invalid register at {:#X}", self.pc);
                self.running = false;
                return;
            },
        }
    }

    fn update_flags(&mut self, result: u32, operand1: u32, operand2: u32, operation: &str) {
        // Zero flag: Set if result is zero
        self.sr.zero = result == 0;

        // Negative flag: Set if the result is negative
        self.sr.negative = (result as i32) < 0;

        match operation {
            "add" => {
                // Carry flag: If the result is less than either operand
                self.sr.carry = result < operand1 || result < operand2;
            }
            "sub" => {
                // Carry flag: If there was a borrow
                self.sr.carry = operand1 > result;
            }
            "mul" => {
                // Carry flag: If the result is greater than the maximum value of u32
                self.sr.carry = result < operand1 || result < operand2;
            }
            _ => {
                // Carry flag: Always false
                self.sr.carry = false;
            }
        }
    }

    fn handle_interrupts(&mut self) {
        if !self.interrupt_enabled || self.interrupt_queue.is_empty() {
            return;
        }

        let interrupt = self.interrupt_queue.remove(0);
        let handler_addr = self.interrupt_vector[interrupt as usize];
        
        self.saved_state = Some(CpuState {
            registers: [self.r0, self.r1, self.r2, self.r3, self.r4, self.r5, self.r6, self.r7],
            pc: self.pc,
            sr: self.sr,
        });

        self.pc = handler_addr;
        self.interrupt_enabled = false;
    }

    fn execute_add(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val.wrapping_add(src_val); // adds the two values

        self.update_flags(result, dest_val, src_val, "add"); // update flags

        self.write_register(dest_reg, result); // set register to result
    }

    fn execute_sub(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val.wrapping_sub(src_val);

        self.update_flags(result, dest_val, src_val, "sub");

        self.write_register(dest_reg, result);
    }

    fn execute_mul(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val.wrapping_mul(src_val);

        self.update_flags(result, dest_val, src_val, "mul");

        self.write_register(dest_reg, result);
    }

    fn execute_div(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);

        if src_val == 0 {
            eprintln!("Division by zero at PC: {:#X}. Halting.", self.pc);
            self.running = false;
            return;
        }

        let result = dest_val.wrapping_div(src_val);

        self.update_flags(result, dest_val, src_val, "div");

        self.write_register(dest_reg, result);
    }

    fn execute_and(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val & src_val;

        self.update_flags(result, dest_val, src_val, "and");

        self.write_register(dest_reg, result);
    }

    fn execute_or(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val | src_val;

        self.update_flags(result, dest_val, src_val, "or");
        self.write_register(dest_reg, result);
    }

    fn execute_xor(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val ^ src_val;

        self.update_flags(result, dest_val, src_val, "xor");
        self.write_register(dest_reg, result);
    }

    fn execute_not(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let result = !src_val;

        self.update_flags(result, src_val, src_val, "not");
        self.write_register(dest_reg, result);
    }

    fn execute_shl(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val << src_val;

        self.update_flags(result, dest_val, src_val, "shl");
        self.write_register(dest_reg, result);
    }

    fn execute_shr(&mut self, src_reg: u8, dest_reg: u8) {
        let src_val = self.read_register(src_reg);
        let dest_val = self.read_register(dest_reg);
        let result = dest_val >> src_val;

        self.update_flags(result, dest_val, src_val, "shr");
        self.write_register(dest_reg, result);
    }

    fn execute_load(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {
        let value = if src_reg == 0xF {
            // Directly use the immediate value if src_reg is 0xF
            immediate as u32
        } else {
            let address = self.read_register(src_reg);
            
            if address as usize >= DATA_MEMORY_SIZE {
                eprintln!("Error: Memory access out of bounds at {:#X}", self.pc);
                self.running = false;
                return;
            }
    
            self.data_memory[address as usize] as u32
        };
    
        self.write_register(dest_reg, value);
    
        self.update_flags(value, 0, 0, "load");
    }

    fn execute_store(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {
        let value = self.read_register(src_reg);
        let address = if dest_reg == 0xF {
            immediate as usize
        } else {
            self.read_register(dest_reg) as usize
        };
    
        if address >= DATA_MEMORY_SIZE {
            eprintln!("Error: Memory access out of bounds at {:#X}", self.pc);
            self.running = false;
            return;
        }
    
        self.data_memory[address] = value as u8;
        self.update_flags(value, 0, 0, "store");
    }
    
    fn execute_mov(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {
        let value = if src_reg == 0xF {
            immediate as u32
        } else {
            self.read_register(src_reg)
        };
        
        self.write_register(dest_reg, value);
        self.update_flags(value, 0, 0, "mov");
    }

    fn execute_jmp(&mut self, immediate: u16) {
        self.pc = immediate as u32;
    }

    fn execute_call(&mut self, immediate: u16) {
        self.call_stack.push(self.pc + 2);
        self.pc = immediate as u32;
    }

    fn execute_ret(&mut self) {
        if let Some(return_addr) = self.call_stack.pop() {
            self.pc = return_addr;
        } else {
            eprintln!("Return stack underflow at {:#X}", self.pc);
        }
    }

    // Helper function for all conditional jumps
    fn execute_conditional_jump(&mut self, condition: bool, immediate: u16) {
        if condition {
            self.pc = immediate as u32;
        } else {
            self.pc += 2;
        }
    }

    fn execute_je(&mut self, immediate: u16) {
        self.execute_conditional_jump(self.sr.zero, immediate);
    }

    fn execute_jne(&mut self, immediate: u16) {
        self.execute_conditional_jump(!self.sr.zero, immediate);
    }

    fn execute_jg(&mut self, immediate: u16) {
        self.execute_conditional_jump(!self.sr.zero && !self.sr.negative, immediate);
    }

    fn execute_jl(&mut self, immediate: u16) {
        self.execute_conditional_jump(self.sr.negative, immediate);
    }

    fn execute_push(&mut self, src_reg: u8, immediate: u16) {
        let value = if src_reg == 0xF {
            immediate as u32
        } else {
            self.read_register(src_reg)
        };

        if self.sp as usize == 0 {
            eprintln!("Stack overflow: SP is at the bottom of the stack at {:#X}", self.pc);
            self.running = false;
            return;
        }

        self.sp -= 4;  // Stack grows downwards (assuming 32-bit values)
        let stack_pos = self.sp as usize;
        let bytes = value.to_le_bytes();  

        self.stack[stack_pos..stack_pos + 4].copy_from_slice(&bytes);
    }

    fn execute_pop(&mut self, dest_reg: u8) {
        if self.sp as usize >= STACK_SIZE {
            eprintln!("Stack underflow: trying to pop from an empty stack at {:#X}", self.pc);
            self.running = false;
            return;
        }

        let stack_pos = self.sp as usize;
        let bytes = &self.stack[stack_pos..stack_pos + 4];
        let value = u32::from_le_bytes(bytes.try_into().expect("Invalid stack data"));

        self.sp += 4;

        self.write_register(dest_reg, value);
    }

    fn execute_int(&mut self, vector: u8) {
        self.interrupt_queue.push(vector);
    }

    fn execute_iret(&mut self) {
        if let Some(state) = self.saved_state.take() {
            [self.r0, self.r1, self.r2, self.r3, self.r4, self.r5, self.r6, self.r7] = state.registers;
            self.pc = state.pc;
            self.sr = state.sr;
            self.interrupt_enabled = true;
        }
    }
}

fn main() { // TODO: Implement loading from file
    let mut cpu = CPU::new();
     
    let program = [
        0x00_0_0_0000,  // NOP
        0x0D_F_0_0003,  // MOV #3 -> R0
        0x0D_F_1_0004,  // MOV #4 -> R1
        0x01_0_1_0000,  // ADD R0 + R1 -> R1
        0x1C_0_0_0000,  // HALT
    ];

    cpu.load_program(&program);
    println!("Initial state:");
    cpu.debug_registers();
    
    cpu.run();
    
    println!("\nFinal state:");
    cpu.debug_registers();
}