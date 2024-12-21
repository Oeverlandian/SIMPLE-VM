use std::env;
use std::{collections::HashMap, vec};

const PROGRAM_MEMORY_SIZE: usize = 1024 * 32; // 32 KB
const DATA_MEMORY_SIZE: usize = 1024 * 8;     // 8 KB
const STACK_SIZE: usize = 1024;               // 1 KB

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
    CALL = 0x0F,   // Call a function or subroutine
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
struct Cpu {
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

impl Cpu {
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

    fn load_program(&mut self, program: Vec<u32>) {
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

            let instruction = self.decode(instruction);
            self.execute(instruction)
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

    fn decode(&mut self, opcode: u32) -> Instruction {
        Instruction::from_opcode(opcode)
    }

    fn execute(&mut self, instruction: Instruction) {

        match instruction.opcode {
            0x00 => { // NOP
                // Do nothing
                self.pc += 4;
            }
            0x01 => { // ADD
                self.execute_add(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
            },
            0x02 => { // SUB
                self.execute_sub(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
            },
            0x03 => { // MUL
                self.execute_mul(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
            }
            0x04 => { // DIV
                self.execute_div(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
            }
            0x05 => { // AND
                self.execute_and(instruction.src_reg, instruction.dest_reg);
                self.pc += 4;
            }
            0x06 => { // OR
                self.execute_or(instruction.src_reg, instruction.dest_reg);
                self.pc += 4;
            }   
            0x07 => { // XOR
                self.execute_xor(instruction.src_reg, instruction.dest_reg);
                self.pc += 4;
            }
            0x08 => { // NOT
                self.execute_not(instruction.src_reg, instruction.dest_reg);
                self.pc += 4;
            }
            0x09 => { // SHL
                self.execute_shl(instruction.src_reg, instruction.dest_reg);
                self.pc += 4;
            }
            0x0A => { // SHR
                self.execute_shr(instruction.src_reg, instruction.dest_reg);
                self.pc += 4;
            }
            0x0B => { // LOAD
                self.execute_load(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
            }
            0x0C => { // STORE
                self.execute_store(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
            },
            0x0D => { // MOV
                self.execute_mov(instruction.src_reg, instruction.dest_reg, instruction.immediate);
                self.pc += 4;
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
            0x15 => { // JGE
                self.execute_jge(instruction.immediate);
            },
            0x16 => { // JLE
                self.execute_jle(instruction.immediate);
            },
            0x17 => { // IN
                self.pc += 4;
            },
            0x18 => { // OUT
                self.pc += 4;
            }
            0x19 => { // PUSH
                self.execute_push(instruction.src_reg, instruction.immediate);
                self.pc += 4;
            },
            0x1A => { // POP
                self.execute_pop(instruction.dest_reg);
                self.pc += 4;
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

    fn execute_add(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {

        let src_val = if src_reg == 0xF { // Checks if the immediate value has to be used
            immediate as u32
        } else {
            self.read_register(src_reg)
        };

        let dest_val = self.read_register(dest_reg);
        let result = dest_val.wrapping_add(src_val); // adds the two values

        self.update_flags(result, dest_val, src_val, "add"); // update flags

        self.write_register(dest_reg, result); // set register to result
    }

    fn execute_sub(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {
        
        let src_val = if src_reg == 0xF {
            immediate as u32
        } else {
            self.read_register(src_reg)
        };

        let dest_val = self.read_register(dest_reg);
        let result = dest_val.wrapping_sub(src_val);

        self.update_flags(result, dest_val, src_val, "sub");

        self.write_register(dest_reg, result);
    }

    fn execute_mul(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {
        
        let src_val = if src_reg == 0xF {
            immediate as u32
        } else {
            self.read_register(src_reg)
        };

        let dest_val = self.read_register(dest_reg);
        let result = dest_val.wrapping_mul(src_val);

        self.update_flags(result, dest_val, src_val, "mul");

        self.write_register(dest_reg, result);
    }

    fn execute_div(&mut self, src_reg: u8, dest_reg: u8, immediate: u16) {

        let src_val = if src_reg == 0xF {
            immediate as u32
        } else {
            self.read_register(src_reg)
        };

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
            self.pc += 4;
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

    fn execute_jge(&mut self, immediate: u16) {
        self.execute_conditional_jump(!self.sr.negative || self.sr.zero, immediate);
    }

    fn execute_jle(&mut self, immediate: u16) {
        self.execute_conditional_jump(self.sr.negative || self.sr.zero , immediate);
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

        self.sp -= 4;
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

#[derive(Debug, PartialEq)]
enum Token {
    Nop,   // No operation
    Add,   // Add two registers
    Sub,   // Subtract two registers
    Mul,   // Multiply two registers
    Div,   // Divide two registers
    And,   // AND between two registers (bitwise)
    Or,    // OR between two registers (bitwise)
    Xor,   // XOR between two registers (bitwise)
    Not,   // NOT between two registers (bitwise)
    Shl,   // Shift left (*2)
    Shr,   // Shift right (/2)
    Load,  // Load data from memory to a register
    Store, // Store data from a register to memory
    Mov,   // Move data between registers
    Jmp,   // Jump to an address
    Call,   // Call a function or subroutine
    Ret,   // Return from function or subroutine
    Je,    // Jump if equal
    Jne,   // Jump if not equal
    Jg,    // Jump if greater than
    Jl,   // Jump if less than
    Jge,   // Jump if greater or equal
    Jle,   // Jump if less or equal
    In,   // Read data from an I/O port
    Out,   // Write data to an I/O port
    Push,  // Push data onto the stack
    Pop,   // Pop data from the stack
    Int,   // Trigger a software interrupt
    Halt,  // Halt the execution
    Iret,  // Returns from an interrupt
    Register(u8),   // E.g. R1
    Immediate(u16), // E.g. #31
    Label(String),  // E.g. loop:
    LabelReference(String), // E.g. @loop
    Eof,
}

#[derive(Debug)]
struct LexerError {
    line_number: usize,
    column: usize,
    message: String,
}

impl std::fmt::Display for LexerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error at line {}, column {}: {}", 
            self.line_number, self.column, self.message)
    }
}

fn create_error(line_number: usize, column: usize, message: &str) -> LexerError {
    LexerError {
        line_number,
        column,
        message: message.to_string(),
    }
}

fn lexer(program: &str) -> Result<Vec<Token>, LexerError>  {
    let mut tokens = vec![];

    for (line_num, line) in program.lines().enumerate() {
        let line = match line.split(';').next() {
            Some(content) => content.trim(),
            None => continue,
        };
        
        if line.is_empty() {
            continue;
        }

        let mut column = 0;
        for token_str in line.split_whitespace() {
            column += token_str.len();
            
            match tokenize_token(token_str) {
                Ok(token) => tokens.push(token),
                Err(msg) => return Err(create_error(line_num + 1, column, &msg)),
            }
        }
    }

    tokens.push(Token::Eof);
    
    Ok(tokens)
}

fn tokenize_token(token: &str) -> Result<Token, String> {
    match token {
        "NOP" => Ok(Token::Nop),
        "ADD" => Ok(Token::Add),
        "SUB" => Ok(Token::Sub),
        "MUL" => Ok(Token::Mul),
        "DIV" => Ok(Token::Div),
        "AND" => Ok(Token::And),
        "OR" => Ok(Token::Or),
        "XOR" => Ok(Token::Xor),
        "NOT" => Ok(Token::Not),
        "SHL" => Ok(Token::Shl),
        "SHR" => Ok(Token::Shr),
        "LOAD" => Ok(Token::Load),
        "STORE" => Ok(Token::Store),
        "MOV" => Ok(Token::Mov),
        "JMP" => Ok(Token::Jmp),
        "CALL" => Ok(Token::Call),
        "RET" => Ok(Token::Ret), 
        "JE" => Ok(Token::Je), 
        "JNE" => Ok(Token::Jne),
        "JG" => Ok(Token::Jg),
        "JL" => Ok(Token::Jl),
        "JGE" => Ok(Token::Jge),
        "JLE" => Ok(Token::Jle),
        "IN" => Ok(Token::In), 
        "OUT" => Ok(Token::Out),
        "PUSH" => Ok(Token::Push),
        "POP" => Ok(Token::Pop),
        "INT" => Ok(Token::Int),
        "HALT" => Ok(Token::Halt),
        "IRET" => Ok(Token::Iret),
        _ => tokenize_others(token),
    }
}

fn tokenize_others(token: &str) -> Result<Token, String> {

    if token.ends_with(':') {
        let label = &token[..token.len()-1];
        if label.is_empty() {
            return Err("Empty label name".to_string());
        }
        if !label.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err("Invalid label name: can only contain alphanumeric characters and underscore".to_string());
        }
        return Ok(Token::Label(label.to_string()));
    }
    
    if token.starts_with('@') {
        let label = &token[1..];
        if label.is_empty() {
            return Err("Empty label reference".to_string());
        }
        if !label.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err("Invalid label reference: can only contain alphanumeric characters and underscore".to_string());
        }
        return Ok(Token::LabelReference(label.to_string()));
    }
    
    if token.starts_with('R') {
        let register_str = token.get(1..)
            .ok_or_else(|| "Invalid register format".to_string())?;
            
        let register = register_str.parse::<u8>()
            .map_err(|_| "Invalid register number".to_string())?;
            
        if register > 7 {
            return Err(format!("Register number {} out of range (0-7)", register));
        }
        
        return Ok(Token::Register(register));
    }
    
    if token.starts_with('#') {
        let immediate_str = token.get(1..)
            .ok_or_else(|| "Invalid immediate format".to_string())?;
            
        let immediate = match immediate_str {
            s if s.starts_with("0x") => u16::from_str_radix(&s[2..], 16)
                .map_err(|_| "Invalid hexadecimal immediate value".to_string())?,
            s if s.starts_with("0b") => u16::from_str_radix(&s[2..], 2)
                .map_err(|_| "Invalid binary immediate value".to_string())?,
            s => s.parse::<u16>()
                .map_err(|_| "Invalid immediate value".to_string())?,
        };
        
        return Ok(Token::Immediate(immediate));
    }
    
    Err(format!("Unrecognized token: {}", token))
}

fn parse(tokens: Vec<Token>) -> Result<Vec<Instruction>, String> {
    let mut instructions = Vec::new();
    let mut label_positions = HashMap::new();
    let mut current = 0;
    let mut position = 0;

    // First pass to collect labels
    while current < tokens.len() && tokens[current] != Token::Eof {
        match &tokens[current] {
            Token::Label(name) => {
                label_positions.insert(name.clone(), position);
            },
            token if is_instruction_start(token) => {
                position += 4;
            },
            _ => {}
        }
        current += 1;
    }

    current = 0;

    // Second pass
    while current < tokens.len() && tokens[current] != Token::Eof {
        match &tokens[current] {
            Token::Label(_) => {
                current += 1;
                continue;
            },
            token if is_instruction_start(token) => {
                match parse_instruction(&tokens, &mut current, &label_positions) {
                    Ok(instruction) => {
                        instructions.push(instruction);
                        position += 4;
                    },
                    Err(e) => return Err(e),
                }
            },
            _ => {
                return Err(format!("Unexpected token: {:?}", tokens[current]));
            }
        }
        current += 1;
    }

    Ok(instructions)
}

fn is_instruction_start(token: &Token) -> bool {
    matches!(token,
        Token::Nop | Token::Add | Token::Sub | Token::Mul | Token::Div | Token::And | Token::Or | Token::Xor | Token::Not | Token::Shl |
        Token::Shr | Token::Load | Token::Store | Token::Mov | Token::Jmp | Token::Call | Token::Ret | Token::Je | Token::Jne | Token::Jg |
        Token::Jl | Token::Jge | Token::Jle | Token::In | Token::Out | Token::Push | Token::Pop | Token::Int | Token::Halt | Token::Iret
    )
}

fn parse_instruction(tokens: &[Token], current: &mut usize, label_positions: &HashMap<String, u32>) -> Result<Instruction, String> {
    match &tokens[*current] {
        Token::Nop => Ok(Instruction {
            opcode: 0x00,
            src_reg: 0,
            dest_reg: 0,
            immediate: 0
        }),

        Token::Add => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in ADD opcode".to_string())
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in ADD instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in ADD instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in ADD instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x01,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Sub => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in SUB opcode".to_string())
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in SUB instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in SUB instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in SUB instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x02,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Mul => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in MUL opcode".to_string())
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in MUL instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in MUL instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in MUL instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x03,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Div => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in DIV opcode".to_string())
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in DIV instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in DIV instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in DIV instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x04,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::And => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in AND opcode".to_string())
            }

            let src_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as source in AND instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in AND instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in AND instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x05,
                src_reg,
                dest_reg,
                immediate: 0,
            })
        },

        Token::Or => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in OR opcode".to_string())
            }

            let src_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as source in OR instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in OR instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in OR instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x06,
                src_reg,
                dest_reg,
                immediate: 0,
            })
        },

        Token::Xor => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in XOR opcode".to_string())
            }

            let src_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as source in XOR instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in XOR instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in XOR instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x07,
                src_reg,
                dest_reg,
                immediate: 0,
            })
        },

        Token::Not => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in NOT opcode".to_string())
            }

            let src_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as source in NOT instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in NOT instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in NOT instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x08,
                src_reg,
                dest_reg,
                immediate: 0,
            })
        },
        
        Token::Shl => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in SHL opcode".to_string())
            }

            let src_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as source in SHL instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in SHL instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in SHL instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0A,
                src_reg,
                dest_reg,
                immediate: 0,
            })
        },

        Token::Shr => {
            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected an operand in SHR opcode".to_string())
            }

            let src_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as source in SHR instruction".to_string()),
            };

            *current += 1;

            if *current >= tokens.len() {
                return Err("Expected a second operand in SHR instruction".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in SHR instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0B,
                src_reg,
                dest_reg,
                immediate: 0,
            })
        },

        Token::Load => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in LOAD opcode".to_string());
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in LOAD instruction".to_string()),
            };

            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected second operand in LOAD instruction".to_string());
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in LOAD instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0B,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Store => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in STORE opcode".to_string());
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in STORE instruction".to_string()),
            };

            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected second operand in STORE instruction".to_string());
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in STORE instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0C,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Mov => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in MOV opcode".to_string());
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in MOV instruction".to_string()),
            };

            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected second operand in MOV instruction".to_string());
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected register as destination in MOV instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0D,
                src_reg,
                dest_reg,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Jmp => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in JMP instruction".to_string());
            }

            let immediate = match &tokens[*current] {
                Token::LabelReference(label) => {
                    *label_positions.get(label)
                        .ok_or_else(|| format!("Undefined label: {}", label))?
                },
                Token::Immediate(imm) => *imm as u32,
                _ => return Err("Expected label reference or immediate value in JMP instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0E,
                src_reg: 0,
                dest_reg: 0,
                immediate: immediate as u16,
            })
        },

        Token::Call => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in CALL instruction".to_string());
            }

            let immediate = match &tokens[*current] {
                Token::LabelReference(label) => {
                    *label_positions.get(label)
                        .ok_or_else(|| format!("Undefined label: {}", label))?
                },
                Token::Immediate(imm) => *imm as u32,
                _ => return Err("Expected label reference or immediate value in CALL instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x0F,
                src_reg: 0,
                dest_reg: 0,
                immediate: immediate as u16,
            })
        },

        

        Token::Je | Token::Jne | Token::Jg | Token::Jl | Token::Jge | Token::Jle => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in conditional jump instruction".to_string());
            }

            let immediate = match &tokens[*current] {
                Token::LabelReference(label) => {
                    *label_positions.get(label)
                        .ok_or_else(|| format!("Undefined label: {}", label))?
                },
                Token::Immediate(imm) => *imm as u32,
                _ => return Err("Expected label reference or immediate value in conditional jump instruction".to_string()),
            };

            let opcode = match &tokens[*current - 1] {
                Token::Je => 0x11,
                Token::Jne => 0x12,
                Token::Jg => 0x13,
                Token::Jl => 0x14,
                Token::Jge => 0x15,
                Token::Jle => 0x16,
                _ => unreachable!(),
            };

            Ok(Instruction {
                opcode,
                src_reg: 0,
                dest_reg: 0,
                immediate: immediate as u16,
            })
        },

        // TODO: ADD IN and OUT after it's implemented (!!)

        Token::Push => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in PUSH opcode".to_string());
            }

            let (src_reg, src_imm) = match &tokens[*current] {
                Token::Register(reg) => (*reg, None),
                Token::Immediate(imm) => (0xF, Some(*imm)),
                _ => return Err("Invalid source operand in PUSH instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x19,
                src_reg,
                dest_reg: 0,
                immediate: src_imm.unwrap_or(0),
            })
        },

        Token::Pop => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in POP opcode".to_string())
            }

            let dest_reg = match &tokens[*current] {
                Token::Register(reg) => *reg,
                _ => return Err("Expected destination register in POP instruction".to_string()),
            };

            Ok(Instruction {
                opcode: 0x1A,
                src_reg: 0,
                dest_reg,
                immediate: 0,
            })
        }

        Token::Int => {
            *current += 1;
            if *current >= tokens.len() {
                return Err("Expected an operand in INT instruction".to_string());
            }

            let immediate = match &tokens[*current] {
                Token::Immediate(imm) => {
                    if *imm > 255 {
                        return Err("Interrupt vector must be between 0 and 255".to_string());
                    }
                    *imm
                },
                _ => return Err("Expected immediate value for interrupt vector".to_string()),
            };

            Ok(Instruction {
                opcode: 0x1B,
                src_reg: 0,
                dest_reg: 0,
                immediate,
            })
        },

        Token::Halt => Ok(Instruction {
            opcode: 0x1C,
            src_reg: 0,
            dest_reg: 0,
            immediate: 0,
        }),

        Token::Iret => Ok(Instruction {
            opcode: 0x1D,
            src_reg: 0,
            dest_reg: 0,
            immediate: 0,
        }),

        _ => Err(format!("Unknown instruction: {:?}", tokens[*current])),
    }
}

fn encode(instructions: Vec<Instruction>) -> Vec<u32> {
    let mut program = vec![];

    for instruction in instructions {
        let instruction = pack_instruction(instruction.opcode, instruction.src_reg, instruction.dest_reg, instruction.immediate);
        program.push(instruction);
    }

    program
}

fn pack_instruction(opcode: u8, src_reg: u8, dest_reg: u8, immediate: u16) -> u32 {
    let opcode = opcode as u32;
    let src_reg = (src_reg & 0xF) as u32;
    let dest_reg = (dest_reg & 0xF) as u32;
    let immediate = immediate as u32;
    
    (opcode << 24) | (src_reg << 20) | (dest_reg << 16) | immediate
}

fn main() {
    let mut cpu = Cpu::new();

    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <file_path>", args[0]);
        return;
    }

    let file_path = &args[1];

    let source_code = match std::fs::read_to_string(file_path) {
        Ok(content) => content,
        Err(err) => {
            println!("Error reading file: {}", err);
            return;
        }
    };
     
    /* let program = vec![
        0x00_0_0_0000,  // NOP
        0x0D_F_0_0003,  // MOV #3 -> R0
        0x0D_F_1_0004,  // MOV #4 -> R1
        0x01_0_1_0000,  // ADD R0 + R1 -> R1
        0x1C_0_0_0000,  // HALT
    ]; */

    /* let program_asm = 
        r#"NOP
        MOV #3 R0
        MOV #4 R1
        ADD R0 R1
        HALT
        "#; */

    let tokens = match lexer(&source_code) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer error: {}", e);
            return;
        }
    };
    
    let instructions = match parse(tokens) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Parser error: {}", e);
            return;
        }
    };

    let program = encode(instructions);

    cpu.debug_registers();
    cpu.load_program(program);
    cpu.run();
    cpu.debug_registers();
    
}