import os
import re
import subprocess

OBJDUMP = "objdump"
X86_AS = "as"
X86_OBJCOPY = "objcopy"

cond_to_suffix = {
    'overflow': 'o',
    'noOverflow': 'no',
    'below': 'b',
    'aboveEqual': 'ae',
    'zero': 'z',
    'notZero': 'nz',
    'belowEqual': 'be',
    'above': 'a',
    'negative': 's',
    'positive': 'ns',
    'parity': 'p',
    'noParity': 'np',
    'less': 'l',
    'greaterEqual': 'ge',
    'lessEqual': 'le',
    'greater': 'g',
}

registers_mapping = {
    # skip rax, rsi, rdi, rsp, rbp as they have special encodings
    # 'rax': {64: 'rax', 32: 'eax', 16: 'ax', 8: 'al'},
    'rcx': {64: 'rcx', 32: 'ecx', 16: 'cx', 8: 'cl'},
    'rdx': {64: 'rdx', 32: 'edx', 16: 'dx', 8: 'dl'},
    'rbx': {64: 'rbx', 32: 'ebx', 16: 'bx', 8: 'bl'},
    # 'rsp': {64: 'rsp', 32: 'esp', 16: 'sp', 8: 'spl'},
    # 'rbp': {64: 'rbp', 32: 'ebp', 16: 'bp', 8: 'bpl'},
    # 'rsi': {64: 'rsi', 32: 'esi', 16: 'si', 8: 'sil'},
    # 'rdi': {64: 'rdi', 32: 'edi', 16: 'di', 8: 'dil'},
    'r8': {64: 'r8', 32: 'r8d', 16: 'r8w', 8: 'r8b'},
    'r9': {64: 'r9', 32: 'r9d', 16: 'r9w', 8: 'r9b'},
    'r10': {64: 'r10', 32: 'r10d', 16: 'r10w', 8: 'r10b'},
    'r11': {64: 'r11', 32: 'r11d', 16: 'r11w', 8: 'r11b'},
    'r12': {64: 'r12', 32: 'r12d', 16: 'r12w', 8: 'r12b'},
    'r13': {64: 'r13', 32: 'r13d', 16: 'r13w', 8: 'r13b'},
    'r14': {64: 'r14', 32: 'r14d', 16: 'r14w', 8: 'r14b'},
    'r15': {64: 'r15', 32: 'r15d', 16: 'r15w', 8: 'r15b'},
    'r16': {64: 'r16', 32: 'r16d', 16: 'r16w', 8: 'r16b'},
    'r17': {64: 'r17', 32: 'r17d', 16: 'r17w', 8: 'r17b'},
    'r18': {64: 'r18', 32: 'r18d', 16: 'r18w', 8: 'r18b'},
    'r19': {64: 'r19', 32: 'r19d', 16: 'r19w', 8: 'r19b'},
    'r20': {64: 'r20', 32: 'r20d', 16: 'r20w', 8: 'r20b'},
    'r21': {64: 'r21', 32: 'r21d', 16: 'r21w', 8: 'r21b'},
    'r22': {64: 'r22', 32: 'r22d', 16: 'r22w', 8: 'r22b'},
    'r23': {64: 'r23', 32: 'r23d', 16: 'r23w', 8: 'r23b'},
    'r24': {64: 'r24', 32: 'r24d', 16: 'r24w', 8: 'r24b'},
    'r25': {64: 'r25', 32: 'r25d', 16: 'r25w', 8: 'r25b'},
    'r26': {64: 'r26', 32: 'r26d', 16: 'r26w', 8: 'r26b'},
    'r27': {64: 'r27', 32: 'r27d', 16: 'r27w', 8: 'r27b'},
    'r28': {64: 'r28', 32: 'r28d', 16: 'r28w', 8: 'r28b'},
    'r29': {64: 'r29', 32: 'r29d', 16: 'r29w', 8: 'r29b'},
    'r30': {64: 'r30', 32: 'r30d', 16: 'r30w', 8: 'r30b'},
    'r31': {64: 'r31', 32: 'r31d', 16: 'r31w', 8: 'r31b'},
}

class Operand(object):
    def generate(self):
        return self

class Register(Operand):
    def generate(self, reg, width):
        self.reg = reg
        self.areg = registers_mapping.get(reg, {}).get(width, reg)
        return self

    def cstr(self):
        return self.reg

    def astr(self):
        return self.areg

class Immediate(Operand):
    def generate(self, value):
        self._value = value
        return self

    def cstr(self):
        return str(self._value)

    def astr(self):
        return str(self._value)

class Address(Operand):
    width_to_ptr = {
        8: "byte ptr",
        16: "word ptr",
        32: "dword ptr",
        64: "qword ptr"
    }

    def generate(self, base, index, width):
        self.base = Register().generate(base, 64)
        self.index = Register().generate(index, 64)
        self._width = width
        return self

    def cstr(self):
        return f"Address({self.base.cstr()}, {self.index.cstr()})"

    def astr(self):
        ptr_str = self.width_to_ptr.get(self._width, "qword ptr")
        return f"{ptr_str} [{self.base.cstr()} + {self.index.cstr()}]"

class Instruction(object):
    def __init__(self, name, aname):
        self._name = name
        self._aname = aname

    def generate_operands(self, *operands):
        self.operands = [operand for operand in operands]

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + ');'

    def astr(self):
        return f'{self._aname} ' + ', '.join([op.astr() for op in self.operands])

class RegInstruction(Instruction):
    def __init__(self, name, aname, width, reg):
        super().__init__(name, aname)
        self.reg = Register().generate(reg, width)
        self.generate_operands(self.reg)

class MemInstruction(Instruction):
    def __init__(self, name, aname, width, mem_base, mem_idx):
        super().__init__(name, aname)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.generate_operands(self.mem)

class TwoRegInstruction(Instruction):
    def __init__(self, name, aname, width, reg1, reg2):
        super().__init__(name, aname)
        self.reg1 = Register().generate(reg1, width)
        self.reg2 = Register().generate(reg2, width)
        self.generate_operands(self.reg1, self.reg2)

    def astr(self):
        return f'{{load}}' + super().astr()

class ThreeRegInstruction(Instruction):
    def __init__(self, name, aname, width, reg1, reg2, reg3):
        super().__init__(name, aname)
        self.reg1 = Register().generate(reg1, width)
        self.reg2 = Register().generate(reg2, width)
        self.reg3 = Register().generate(reg3, width)
        self.generate_operands(self.reg1, self.reg2, self.reg3)

class MemRegInstruction(Instruction):
    def __init__(self, name, aname, width, reg, mem_base, mem_idx):
        super().__init__(name, aname)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.reg = Register().generate(reg, width)
        self.generate_operands(self.mem, self.reg)

class RegMemInstruction(Instruction):
    def __init__(self, name, aname, width, reg, mem_base, mem_idx):
        super().__init__(name, aname)
        self.reg = Register().generate(reg, width)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.generate_operands(self.reg, self.mem)

class RegImmInstruction(Instruction):
    def __init__(self, name, aname, width, reg, imm):
        super().__init__(name, aname)
        self.reg = Register().generate(reg, width)
        self.imm = Immediate().generate(imm)
        self.generate_operands(self.reg, self.imm)

class MemImmInstruction(Instruction):
    def __init__(self, name, aname, width, imm, mem_base, mem_idx):
        super().__init__(name, aname)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.imm = Immediate().generate(imm)
        self.generate_operands(self.mem, self.imm)

class RegRegImmInstruction(Instruction):
    def __init__(self, name, aname, width, reg1, reg2, imm):
        super().__init__(name, aname)
        self.reg1 = Register().generate(reg1, width)
        self.reg2 = Register().generate(reg2, width)
        self.imm = Immediate().generate(imm)
        self.generate_operands(self.reg1, self.reg2, self.imm)

class RegMemImmInstruction(Instruction):
    def __init__(self, name, aname, width, reg, imm, mem_base, mem_idx):
        super().__init__(name, aname)
        self.reg = Register().generate(reg, width)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.imm = Immediate().generate(imm)
        self.generate_operands(self.reg, self.mem, self.imm)

class RegMemRegInstruction(Instruction):
    def __init__(self, name, aname, width, reg1, mem_base, mem_idx, reg2):
        super().__init__(name, aname)
        self.reg1 = Register().generate(reg1, width)
        self.reg2 = Register().generate(reg2, width)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.generate_operands(self.reg1, self.mem, self.reg2)

class RegRegMemInstruction(Instruction):
    def __init__(self, name, aname, width, reg1, reg2, mem_base, mem_idx):
        super().__init__(name, aname)
        self.reg1 = Register().generate(reg1, width)
        self.reg2 = Register().generate(reg2, width)
        self.mem = Address().generate(mem_base, mem_idx, width)
        self.generate_operands(self.reg1, self.reg2, self.mem)

class RegRegRegImmInstruction(Instruction):
    def __init__(self, name, aname, width, reg1, reg2, reg3, imm):
        super().__init__(name, aname)
        self.reg1 = Register().generate(reg1, width)
        self.reg2 = Register().generate(reg2, width)
        self.reg3 = Register().generate(reg3, width)
        self.imm = Immediate().generate(imm)
        self.generate_operands(self.reg1, self.reg2, self.reg3, self.imm)

class Pop2Instruction(TwoRegInstruction):
    def __init__(self, name, aname, width, reg1, reg2):
        super().__init__(name, aname, width, reg1, reg2)

    def cstr(self):
        # reverse to match the order in OpenJDK
        return f'__ {self._name} (' + ', '.join([reg.cstr() for reg in reversed(self.operands)]) + ');'

class Push2Instruction(TwoRegInstruction):
    def __init__(self, name, aname, width, reg1, reg2):
        super().__init__(name, aname, width, reg1, reg2)

    def cstr(self):
        # reverse to match the order in OpenJDK
        return f'__ {self._name} (' + ', '.join([reg.cstr() for reg in reversed(self.operands)]) + ');'

class CondRegMemInstruction(RegMemInstruction):
    def __init__(self, name, aname, width, cond, reg, mem_base, mem_idx):
        super().__init__(name, aname, width, reg, mem_base, mem_idx)
        self.cond = cond

    def cstr(self):
        return f'__ {self._name} (' + 'Assembler::Condition::' + self.cond + ', ' + ', '.join([self.reg.cstr(), self.mem.cstr()]) + ');'

    def astr(self):
        return f'{self._aname}' + cond_to_suffix[self.cond] + ' ' + ', '.join([self.reg.astr(), self.mem.astr()])

class CondRegInstruction(RegInstruction):
    def __init__(self, name, aname, width, cond, reg):
        super().__init__(name, aname, width, reg)
        self.cond = cond

    def cstr(self):
        return f'__ {self._name}b (' + 'Assembler::Condition::' + self.cond + ', ' + self.reg.cstr() + ');'

    def astr(self):
        return f'{self._aname}' + cond_to_suffix[self.cond] + ' ' + self.reg.astr()
class RegMemNddInstruction(RegMemInstruction):
    def __init__(self, name, aname, width, no_flag, reg, mem_base, mem_idx):
        super().__init__(name, aname, width, reg, mem_base, mem_idx)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        cl_str = (', cl' if (self._name == 'eroll' or self._name == 'erolq' or self._name == 'erorl' or self._name == 'erorq' or 
                             self._name == 'esall' or self._name == 'esalq' or self._name == 'esarq' or self._name == 'esarl' or
                             self._name == 'eshrl' or self._name == 'eshrq') else '')
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + ', '.join([self.reg.astr(), self.mem.astr()]) + cl_str

class RegMemImmNddInstruction(RegMemImmInstruction):
    def __init__(self, name, aname, width, no_flag, reg, imm, mem_base, mem_idx):
        super().__init__(name, aname, width, reg, imm, mem_base, mem_idx)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + ', '.join([self.reg.astr(), self.mem.astr(), self.imm.astr()])

class RegMemRegNddInstruction(RegMemRegInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, mem_base, mem_idx, reg2):
        super().__init__(name, aname, width, reg1, mem_base, mem_idx, reg2)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        return ('{NF}' if self.no_flag else '') + f'{self._aname} ' + ', '.join([self.reg1.astr(), self.mem.astr(), self.reg2.astr()])

class RegRegImmNddInstruction(RegRegImmInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, reg2, imm):
        super().__init__(name, aname, width, reg1, reg2, imm)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + ', '.join([self.reg1.astr(), self.reg2.astr(), self.imm.astr()])

class RegRegMemNddInstruction(RegRegMemInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, reg2, mem_base, mem_idx):
        super().__init__(name, aname, width, reg1, reg2, mem_base, mem_idx)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?
    
    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        return ('{NF}' if self.no_flag else '') + f'{self._aname} ' + ', '.join([self.reg1.astr(), self.reg2.astr(), self.mem.astr()])
    
class RegRegNddInstruction(TwoRegInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, reg2):
        super().__init__(name, aname, width, reg1, reg2)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        # TODO: find document
        cl_str = (', cl' if (self._name == 'eroll' or self._name == 'erolq' or self._name == 'erorl' or self._name == 'erorq' or 
                             self._name == 'esall' or self._name == 'esalq' or self._name == 'esarl' or self._name == 'esarq' or 
                             self._name == 'eshll' or self._name == 'eshlq' or self._name == 'eshrl' or self._name == 'eshrq') else '')
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + ', '.join([reg.astr() for reg in self.operands]) + cl_str

class RegRegRegNddInstruction(ThreeRegInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, reg2, reg3):
        super().__init__(name, aname, width, reg1, reg2, reg3)
        self.no_flag = no_flag # TODO: noflag=False == noflag=None?

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        cl_str = (', cl' if (self._name == 'eshldl' or self._name == 'eshldq' or self._name == 'eshrdl' or self._name == 'eshrdq') else '')
        return ('{NF}' if self.no_flag else '') + f'{self._aname} ' + ', '.join([self.reg1.astr(), self.reg3.astr(), self.reg2.astr()]) + cl_str

class CondRegRegRegNddInstruction(ThreeRegInstruction):
    def __init__(self, name, aname, width, cond, reg1, reg2, reg3):
        super().__init__(name, aname, width, reg1, reg2, reg3)
        self.cond = cond
    
    def cstr(self):
        return f'__ {self._name} (' + 'Assembler::Condition::' + self.cond + ', ' + ', '.join([reg.cstr() for reg in self.operands]) + ');'
    
    def astr(self):
        return f'{self._aname}' + cond_to_suffix[self.cond] + ' ' + ', '.join([reg.astr() for reg in self.operands])

class CondRegRegMemNddInstruction(RegRegMemInstruction):
    def __init__(self, name, aname, width, cond, reg1, reg2, mem_base, mem_idx):
        super().__init__(name, aname, width, reg1, reg2, mem_base, mem_idx)
        self.cond = cond
    
    def cstr(self):
        return f'__ {self._name} (' + 'Assembler::Condition::' + self.cond + ', ' + ', '.join([reg.cstr() for reg in self.operands]) + ');'
    
    def astr(self):
        return f'{self._aname}' + cond_to_suffix[self.cond] + ' ' + ', '.join([reg.astr() for reg in self.operands])

class RegNddInstruction(RegInstruction):
    def __init__(self, name, aname, width, no_flag, reg):
        super().__init__(name, aname, width, reg)
        self.no_flag = no_flag
    
    def cstr(self):
        return f'__ {self._name} (' + self.reg.cstr() + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'
    
    def astr(self):
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + self.reg.astr()

class MemNddInstruction(MemInstruction):
    def __init__(self, name, aname, width, no_flag, mem_base, mem_idx):
        super().__init__(name, aname, width, mem_base, mem_idx)
        self.no_flag = no_flag
    
    def cstr(self):
        return f'__ {self._name} (' + self.mem.cstr() + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'
    
    def astr(self):
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + self.mem.astr()

class RegRegRegImmNddInstruction(RegRegRegImmInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, reg2, reg3, imm):
        super().__init__(name, aname, width, reg1, reg2, reg3, imm)
        self.no_flag = no_flag
        
    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'
    
    def astr(self):
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + ', '.join([self.reg1.astr(), self.reg2.astr(), self.reg3.astr(), self.imm.astr()])

class RegImm32Instruction(RegImmInstruction):
    def __init__(self, name, aname, width, reg, imm):
        super().__init__(name, aname, width, reg, imm)

    def cstr(self):
        return f'__ {self._name} (' + ', '.join([self.reg.cstr(), self.imm.cstr()]) + ');'

class RegRegImm32NddInstruction(RegRegImmInstruction):
    def __init__(self, name, aname, width, no_flag, reg1, reg2, imm):
        super().__init__(name, aname, width, reg1, reg2, imm)
        self.no_flag = no_flag

    def cstr(self):
        return f'__ {self._name}(' + ', '.join([op.cstr() for op in self.operands]) + (f', {str(self.no_flag).lower()}' if self.no_flag is not None else '') + ');'

    def astr(self):
        return ('{NF}' if self.no_flag else '{EVEX}') + f'{self._aname} ' + ', '.join([self.reg1.astr(), self.reg2.astr(), self.imm.astr()])

instrs = []
test_regs = list(registers_mapping.keys())

immediates32 = [2 ** i for i in range(0, 32, 4)]
immediates16 = [2 ** i for i in range(0, 16, 2)]
immediates8 = [2 ** i for i in range(0, 8, 2)]
immediates5 = [2 ** i for i in range(0, 5, 1)]
immediate_values_8_to_16_bit = [2 ** i for i in range(8, 16, 2)]
immediate_values_16_to_32_bit = [2 ** i for i in range(16, 32, 2)]

immediate_map = {
    8: immediates8,
    16: immediates16,
    32: immediates32,
    64: immediates32
}

ifdef_flags = []

def is_64_reg(reg):
    return reg in {'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'}

def print_instruction(instr, lp64_flag, print_lp64_flag):
    cstr = instr.cstr()
    astr = instr.astr()
    print("    %-50s //    %s" % (cstr, astr))
    ifdef_flags.append(lp64_flag or not print_lp64_flag)
    instrs.append(cstr)
    outfile.write(f"    {astr}\n")

def handle_lp64_flag(i, lp64_flag, print_lp64_flag):
    if is_64_reg(test_regs[i]) and not lp64_flag and print_lp64_flag:
        print("#ifdef _LP64")
        return True
    return lp64_flag

def get_immediate_list(op_name, width):
    # special cases
    # TODO: combine with cl cases
    shift_ops = {'sarl', 'sarq', 'shll', 'shlq', 'shrl', 'shrq', 'shrdl', 'shrdq', 'shldl', 'shldq', 'rcrq', 'rorl', 'rorq', 'roll', 'rolq', 'rcll', 'rclq',
                 'esarl', 'esarq', 'eshll', 'eshlq', 'eshrl', 'eshrq', 'eshrdl', 'eshrdq', 'eshldl', 'eshldq', 'ercrq', 'erorl', 'erorq', 'eroll', 'erolq', 'ercll', 'erclq',
                 'esall', 'esalq'}
    addw_ops = {'addw'}
    if op_name in shift_ops:
        return immediates5
    elif op_name in addw_ops:
        return immediate_values_8_to_16_bit
    else:
        return immediate_map[width]

def generate(RegOp, ops, print_lp64_flag=True):
    for op in ops:
        op_name = op[0]
        width = op[2]
        lp64_flag = False

        if RegOp in [RegInstruction, CondRegInstruction, RegNddInstruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag(i, lp64_flag, print_lp64_flag)
                instr = RegOp(*op, reg=test_regs[i])
                print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [TwoRegInstruction, RegRegNddInstruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 1) % len(test_regs), lp64_flag, print_lp64_flag)
                instr = RegOp(*op, reg1=test_regs[i], reg2=test_regs[(i + 1) % len(test_regs)])
                print_instruction(instr, lp64_flag, print_lp64_flag)
        
        elif RegOp in [ThreeRegInstruction, RegRegRegNddInstruction, CondRegRegRegNddInstruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 2) % len(test_regs), lp64_flag, print_lp64_flag)
                instr = RegOp(*op, reg1=test_regs[i], reg2=test_regs[(i + 1) % len(test_regs)], reg3=test_regs[(i + 2) % len(test_regs)])
                print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [MemRegInstruction, RegMemInstruction, CondRegMemInstruction, RegMemNddInstruction]:
            for i in range(len(test_regs)):
                if test_regs[(i + 2) % len(test_regs)] == 'rsp':
                    continue
                lp64_flag = handle_lp64_flag((i + 2) % len(test_regs), lp64_flag, print_lp64_flag)
                instr = RegOp(*op, reg=test_regs[i], mem_base=test_regs[(i + 1) % len(test_regs)], mem_idx=test_regs[(i + 2) % len(test_regs)])
                print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [RegImmInstruction]:
            imm_list = get_immediate_list(op_name, width)
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag(i, lp64_flag, print_lp64_flag)
                for imm in imm_list:
                    instr = RegOp(*op, reg=test_regs[i], imm=imm)
                    print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [MemImmInstruction]:
            imm_list = get_immediate_list(op_name, width)
            for imm in imm_list:
                for i in range(len(test_regs)):
                    if test_regs[(i + 1) % len(test_regs)] == 'rsp':
                        continue
                    lp64_flag = handle_lp64_flag((i + 1) % len(test_regs), lp64_flag, print_lp64_flag)
                    instr = RegOp(*op, imm=imm, mem_base=test_regs[i], mem_idx=test_regs[(i + 1) % len(test_regs)])
                    print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [MemInstruction, MemNddInstruction]:
            for i in range(len(test_regs)):
                if test_regs[(i + 1) % len(test_regs)] == 'rsp':
                    continue
                lp64_flag = handle_lp64_flag((i + 1) % len(test_regs), lp64_flag, print_lp64_flag)
                instr = RegOp(*op, mem_base=test_regs[i], mem_idx=test_regs[(i + 1) % len(test_regs)])
                print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [RegRegImmInstruction, RegRegImmNddInstruction]:
            imm_list = get_immediate_list(op_name, width)
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 1) % len(test_regs), lp64_flag, print_lp64_flag)
                for imm in imm_list:
                    instr = RegOp(*op, reg1=test_regs[i], reg2=test_regs[(i + 1) % len(test_regs)], imm=imm)
                    print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [RegMemImmInstruction, RegMemImmNddInstruction]:
            imm_list = get_immediate_list(op_name, width)
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 2) % len(test_regs), lp64_flag, print_lp64_flag)
                for imm in imm_list:
                    if test_regs[(i + 2) % len(test_regs)] == 'rsp':
                        continue
                    instr = RegOp(*op, reg=test_regs[i], mem_base=test_regs[(i + 1) % len(test_regs)], mem_idx=test_regs[(i + 2) % len(test_regs)], imm=imm)
                    print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [RegMemRegInstruction, RegRegMemInstruction, RegMemRegNddInstruction, RegRegMemNddInstruction, CondRegRegMemNddInstruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 2) % len(test_regs), lp64_flag, print_lp64_flag)
                if test_regs[(i + 2) % len(test_regs)] == 'rsp':
                    continue
                instr = RegOp(*op, reg1=test_regs[i], mem_base=test_regs[(i + 1) % len(test_regs)], mem_idx=test_regs[(i + 2) % len(test_regs)], reg2=test_regs[(i + 3) % len(test_regs)])
                print_instruction(instr, lp64_flag, print_lp64_flag)
        
        elif RegOp in [RegRegRegImmInstruction, RegRegRegImmNddInstruction]:
            imm_list = get_immediate_list(op_name, width)
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 2) % len(test_regs), lp64_flag, print_lp64_flag)
                for imm in imm_list:
                    instr = RegOp(*op, reg1=test_regs[i], reg2=test_regs[(i + 1) % len(test_regs)], reg3=test_regs[(i + 2) % len(test_regs)], imm=imm)
                    print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [Push2Instruction, Pop2Instruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 1) % len(test_regs), lp64_flag, print_lp64_flag)
                if test_regs[(i + 1) % len(test_regs)] == 'rsp' or test_regs[i] == 'rsp':
                    continue
                instr = RegOp(*op, reg1=test_regs[i], reg2=test_regs[(i + 1) % len(test_regs)])
                print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [RegImm32Instruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag(i, lp64_flag, print_lp64_flag)
                for imm in immediate_values_16_to_32_bit:
                    instr = RegOp(*op, reg=test_regs[i], imm=imm)
                    print_instruction(instr, lp64_flag, print_lp64_flag)

        elif RegOp in [RegRegImm32NddInstruction]:
            for i in range(len(test_regs)):
                lp64_flag = handle_lp64_flag((i + 1) % len(test_regs), lp64_flag, print_lp64_flag)
                for imm in immediate_values_16_to_32_bit:
                    instr = RegOp(*op, reg1=test_regs[i], reg2=test_regs[(i + 1) % len(test_regs)], imm=imm)
                    print_instruction(instr, lp64_flag, print_lp64_flag)
        
        else:
            raise ValueError(f"Unsupported instruction type: {RegOp}")

        if lp64_flag and print_lp64_flag:
            print("#endif // _LP64")
            lp64_flag = False

def print_with_ifdef(ifdef_flags, items, item_formatter, items_per_line=1):
    under_defined = False
    current_line_length = 0
    for idx, item in enumerate(items):
        if ifdef_flags[idx]:
            if not under_defined:
                if current_line_length > 0:
                    print()
                print("#ifdef _LP64")
                under_defined = True
                current_line_length = 0
        else:
            if under_defined:
                if current_line_length > 0:
                    print()
                print("#endif // _LP64")
                under_defined = False
                current_line_length = 0
        if current_line_length == 0:
            print("   ", end="")
        print(f" {item_formatter(item)},", end="")
        current_line_length += 1
        if idx % items_per_line == items_per_line - 1:
            print()
            current_line_length = 0
    if under_defined:
        if current_line_length > 0:
            print()
        print("#endif // _LP64")

print("// BEGIN  Generated code -- do not edit")
print("// Generated by x86-asmtest.py")

outfile = open("x86ops.s", "w")
outfile.write(".intel_syntax noprefix\n")

instruction_set = {
    TwoRegInstruction: [
    #     ('shldl', 'shld', 32),
    #     ('shrdl', 'shrd', 32),
    #     ('adcl', 'adc', 32),
    #     ('imull', 'imul', 32),
    #     ('popcntl', 'popcnt', 32),
    #     ('sbbl', 'sbb', 32),
    #     ('subl', 'sub', 32),
    #     ('tzcntl', 'tzcnt', 32),
    #     ('lzcntl', 'lzcnt', 32),
    #     ('addl', 'add', 32),
    #     ('andl', 'and', 32),
    #     ('orl', 'or', 32),
        # ('xorl', 'xor', 32),
    ],
    # MemRegInstruction: [
        # ('addb', 'add', 8),
    #     ('addw', 'add', 16),
    #     ('addl', 'add', 32),
    #     ('adcl', 'adc', 32),
    #     ('andb', 'and', 8),
    #     ('andl', 'and', 32),
    #     ('orb', 'or', 8),
    #     ('orl', 'or', 32),
    #     ('xorb', 'xor', 8),
    #     ('xorl', 'xor', 32),
    #     ('subl', 'sub', 32),
    # ],
    # MemImmInstruction: [
    #     ('adcl', 'adc', 32),
    #     ('andl', 'and', 32),
    #     ('addb', 'add', 8),
    #     ('addw', 'add', 16),
    #     ('addl', 'add', 32),
    #     ('sarl', 'sar', 32),
    #     ('sbbl', 'sbb', 32),
    #     ('shrl', 'shr', 32),
    #     ('subl', 'sub', 32),
    #     ('xorl', 'xor', 32),
    #     ('orb', 'or', 8),
    #     ('orl', 'or', 32),
    # ],
    # RegMemInstruction: [
    #     ('addl', 'add', 32),
    #     ('andl', 'and', 32),
    #     ('lzcntl', 'lzcnt', 32),
    #     ('orl', 'or', 32),
    #     ('adcl', 'adc', 32),
    #     ('imull', 'imul', 32),
    #     ('popcntl', 'popcnt', 32),
    #     ('sbbl', 'sbb', 32),
    #     ('subl', 'sub', 32),
    #     ('tzcntl', 'tzcnt', 32),
    #     ('xorb', 'xor', 8),
    #     ('xorw', 'xor', 16),
    #     ('xorl', 'xor', 32),
    # ],
    # RegImmInstruction: [
    #     ('addb', 'add', 8),
    #     ('addl', 'add', 32),
    #     ('andl', 'and', 32),
    #     ('adcl', 'adc', 32),
    #     ('rcll', 'rcl', 32),
    #     ('roll', 'rol', 32),
    #     ('rorl', 'ror', 32),
    #     ('sarl', 'sar', 32),
    #     ('sbbl', 'sbb', 32),
    #     ('shll', 'shl', 32),
    #     ('shrl', 'shr', 32),
    #     ('subl', 'sub', 32),
    #     ('xorl', 'xor', 32),
    # ],
    # CondRegMemInstruction: [
    #     ('cmovl', 'cmov', 32, key) for key in cond_to_suffix.keys()
    # ],
    # CondRegInstruction: [
    #     ('set', 'set', 8, key) for key in cond_to_suffix.keys()
    # ],
    # RegInstruction: [
    #     ('divl', 'div', 32),
    #     ('idivl', 'idiv', 32),
    #     ('imull', 'imul', 32),
    #     ('mull', 'mul', 32),
    #     ('negl', 'neg', 32),
    #     ('notl', 'not', 32),
    #     ('roll', 'rol', 32),
    #     ('rorl', 'ror', 32),
    #     ('sarl', 'sar', 32),
    #     ('shll', 'shl', 32),
    #     ('shrl', 'shr', 32),
    #     ('incrementl', 'inc', 32),
    #     ('decrementl', 'dec', 32),
    # ],
    # MemInstruction: [
    #     ('mull', 'mul', 32),
    #     ('negl', 'neg', 32),
    #     ('sarl', 'sar', 32),
    #     ('shrl', 'shr', 32),
    #     ('incrementl', 'inc', 32),
    #     ('decrementl', 'dec', 32),
    # ],
    # RegMemImmInstruction: [
    #     ('imull', 'imul', 32),
    # ],
    # RegRegImmInstruction: [
    #     ('imull', 'imul', 32),
    #     ('shldl', 'shld', 32),
    #     ('shrdl', 'shrd', 32),
    # ],
    # RegImm32Instruction: [
    #     ('subl_imm32', 'sub', 32),
    # ],
    # --- NDD instructions ---
    RegNddInstruction: [
        ('eidivl', 'idiv', 32, False),
        ('eidivl', 'idiv', 32, True),
        ('edivl', 'div', 32, False),
        ('edivl', 'div', 32, True),
        ('eimull', 'imul', 32, False),
        ('eimull', 'imul', 32, True),
        ('emull', 'mul', 32, False),
        ('emull', 'mul', 32, True),
    ],
    MemNddInstruction: [
        ('emull', 'mul', 32, False),
        ('emull', 'mul', 32, True),
    ],
    RegRegNddInstruction: [
        ('elzcntl', 'lzcnt', 32, False),
        ('elzcntl', 'lzcnt', 32, True),
        ('enegl', 'neg', 32, False),
        ('enegl', 'neg', 32, True),
        ('enotl', 'not', 32, None),
        ('eroll', 'rol', 32, False),
        ('eroll', 'rol', 32, True),
        ('erorl', 'ror', 32, False),
        ('erorl', 'ror', 32, True),
        ('esall', 'sal', 32, False),
        ('esall', 'sal', 32, True),
        ('esarl', 'sar', 32, False),
        ('esarl', 'sar', 32, True),
        ('edecl', 'dec', 32, False), # protected
        ('edecl', 'dec', 32, True), # protected
        ('eincl', 'inc', 32, False), # protected
        ('eincl', 'inc', 32, True), # protected 
        ('eshll', 'shl', 32, False),
        ('eshll', 'shl', 32, True),
        ('eshrl', 'shr', 32, False),
        ('eshrl', 'shr', 32, True),
        ('etzcntl', 'tzcnt', 32, False),
        ('etzcntl', 'tzcnt', 32, True),
    ],
    RegMemNddInstruction: [
        ('elzcntl', 'lzcnt', 32, False),
        ('elzcntl', 'lzcnt', 32, True),
        ('enegl', 'neg', 32, False),
        ('enegl', 'neg', 32, True),
        ('esall', 'sal', 32, False),
        ('esall', 'sal', 32, True),
        ('esarl', 'sar', 32, False),
        ('esarl', 'sar', 32, True),
        ('edecl', 'dec', 32, False), # protected
        ('edecl', 'dec', 32, True), # protected
        ('eincl', 'inc', 32, False), # protected
        ('eincl', 'inc', 32, True), # protected
        ('eshrl', 'shr', 32, False),
        ('eshrl', 'shr', 32, True),
        ('etzcntl', 'tzcnt', 32, False),
        ('etzcntl', 'tzcnt', 32, True),
    ],
    RegMemImmNddInstruction: [
        ('eaddl', 'add', 32, False),
        ('eaddl', 'add', 32, True),
        ('eandl', 'and', 32, False),
        ('eandl', 'and', 32, True),
        # ('eimull', 'imul', 32, False), # 1c -> 0c, ND bit 
        # ('eimull', 'imul', 32, True), # 1c -> 0c, ND bit
        ('eorl', 'or', 32, False),
        ('eorl', 'or', 32, True),
        ('eorb', 'or', 8, False),
        ('eorb', 'or', 8, True),
        ('esall', 'sal', 32, False),
        ('esall', 'sal', 32, True),
        ('esarl', 'sar', 32, False),
        ('esarl', 'sar', 32, True),
        ('eshrl', 'shr', 32, False),
        ('eshrl', 'shr', 32, True),
        ('esubl', 'sub', 32, False),
        ('esubl', 'sub', 32, True),
        ('exorl', 'xor', 32, False),
        ('exorl', 'xor', 32, True),
    ],
    RegMemRegNddInstruction: [
        ('eaddl', 'add', 32, False),
        ('eaddl', 'add', 32, True),
        ('eorl', 'or', 32, False),
        ('eorl', 'or', 32, True),
        ('eorb', 'or', 8, False),
        ('eorb', 'or', 8, True),
        ('esubl', 'sub', 32, False),
        ('esubl', 'sub', 32, True),
        ('exorl', 'xor', 32, False),
        ('exorl', 'xor', 32, True),
        ('exorb', 'xor', 8, False),
        ('exorb', 'xor', 8, True),
    ],
    RegRegImmNddInstruction: [
        ('eaddl', 'add', 32, False),
        ('eaddl', 'add', 32, True),
        ('eandl', 'and', 32, False),
        ('eandl', 'and', 32, True),
        ('eimull', 'imul', 32, False),
        ('eimull', 'imul', 32, True),
        ('eorl', 'or', 32, False),
        ('eorl', 'or', 32, True),
        ('ercll', 'rcl', 32, None),
        ('eroll', 'rol', 32, False),
        ('eroll', 'rol', 32, True),
        ('erorl', 'ror', 32, False),
        ('erorl', 'ror', 32, True),
        ('esall', 'sal', 32, False),
        ('esall', 'sal', 32, True),
        ('esarl', 'sar', 32, False),
        ('esarl', 'sar', 32, True),
        ('eshll', 'shl', 32, False),
        ('eshll', 'shl', 32, True),
        ('eshrl', 'shr', 32, False),
        ('eshrl', 'shr', 32, True),
        ('esubl', 'sub', 32, False),
        ('esubl', 'sub', 32, True),
        ('exorl', 'xor', 32, False),
        ('exorl', 'xor', 32, True),
    ],
    RegRegMemNddInstruction: [
        ('eaddl', 'add', 32, False),
        ('eaddl', 'add', 32, True),
        ('eandl', 'and', 32, False),
        ('eandl', 'and', 32, True),
        ('eimull', 'imul', 32, False),
        ('eimull', 'imul', 32, True),
        ('eorl', 'or', 32, False),
        ('eorl', 'or', 32, True),
        ('esubl', 'sub', 32, False),
        ('esubl', 'sub', 32, True),
        ('exorl', 'xor', 32, False),
        ('exorl', 'xor', 32, True),
        ('exorb', 'xor', 8, False),
        ('exorb', 'xor', 8, True),
        # ('exorw', 'xor', 16, False), # add 66 prefix
        # ('exorw', 'xor', 16, True), # add 66 prefix
    ],
    RegRegRegNddInstruction: [
        ('eaddl', 'add', 32, False),
        ('eaddl', 'add', 32, True),
        ('eandl', 'and', 32, False),
        ('eandl', 'and', 32, True),
        # ('eimull', 'imul', 32, False), # reverse
        # ('eimull', 'imul', 32, True), # reverse
        # ('eorw', 'or', 16, False), # 01 at pp bits
        # ('eorw', 'or', 16, True), # 01 at pp bits
        ('eorl', 'or', 32, False),
        ('eorl', 'or', 32, True),
        ('eshldl', 'shld', 32, False),
        ('eshldl', 'shld', 32, True),
        ('eshrdl', 'shrd', 32, False),
        ('eshrdl', 'shrd', 32, True),
        # ('esubl', 'sub', 32, False), # reverse
        # ('esubl', 'sub', 32, True), # reverse
        ('exorl', 'xor', 32, False),
        ('exorl', 'xor', 32, True),
    ],
    RegRegRegImmNddInstruction: [
        # ('eshldl', 'shld', 32, False), # reverse
        # ('eshldl', 'shld', 32, True), # reverse
        # ('eshrdl', 'shrd', 32, False), # reverse
        # ('eshrdl', 'shrd', 32, True), # reverse
    ],
    CondRegRegRegNddInstruction: [
        ('ecmovl', 'cmov', 32, key) for key in cond_to_suffix.keys()
    ],
    CondRegRegMemNddInstruction: [
        ('ecmovl', 'cmov', 32, key) for key in cond_to_suffix.keys()
    ],
    RegRegImm32NddInstruction: [
        ('esubl_imm32', 'sub', 32, False),
        ('esubl_imm32', 'sub', 32, True),
    ],
}

instruction_set64 = {
    # TwoRegInstruction: [
    #     ('adcq', 'adc', 64),
    #     ('imulq', 'imul', 64),
    #     ('popcntq', 'popcnt', 64),
    #     ('sbbq', 'sbb', 64),
    #     ('subq', 'sub', 64),
    #     ('tzcntq', 'tzcnt', 64),
    #     ('lzcntq', 'lzcnt', 64),
    #     ('addq', 'add', 64),
    #     ('andq', 'and', 64),
    #     ('orq', 'or', 64),
    #     ('xorq', 'xor', 64)
    # ],
    # MemRegInstruction: [
    #     ('addq', 'add', 64),
    #     ('andq', 'and', 64),
    #     ('orq', 'or', 64),
    #     ('xorq', 'xor', 64),
    #     ('subq', 'sub', 64)
    # ],
    # MemImmInstruction: [
    #     ('andq', 'and', 64),
    #     ('addq', 'add', 64),
    #     ('sarq', 'sar', 64),
    #     ('sbbq', 'sbb', 64),
    #     ('shrq', 'shr', 64),
    #     ('subq', 'sub', 64),
    #     ('xorq', 'xor', 64),
    #     ('orq', 'or', 64)
    # ],
    # RegMemInstruction: [
    #     ('addq', 'add', 64),
    #     ('andq', 'and', 64),
    #     ('lzcntq', 'lzcnt', 64),
    #     ('orq', 'or', 64),
    #     ('adcq', 'adc', 64),
    #     ('imulq', 'imul', 64),
    #     ('popcntq', 'popcnt', 64),
    #     ('sbbq', 'sbb', 64),
    #     ('subq', 'sub', 64),
    #     ('tzcntq', 'tzcnt', 64),
    #     ('xorq', 'xor', 64)
    # ],
    # RegImmInstruction: [
    #     ('addq', 'add', 64),
    #     ('andq', 'and', 64),
    #     ('adcq', 'adc', 64),
    #     ('rclq', 'rcl', 64),
    #     ('rcrq', 'rcr', 64),
    #     ('rolq', 'rol', 64),
    #     ('rorq', 'ror', 64),
    #     ('sarq', 'sar', 64),
    #     ('sbbq', 'sbb', 64),
    #     ('shlq', 'shl', 64),
    #     ('shrq', 'shr', 64),
    #     ('subq', 'sub', 64),
    #     ('xorq', 'xor', 64),
    # ],
    # CondRegMemInstruction: [
    #     ('cmovq', 'cmov', 64, key) for key in cond_to_suffix.keys()
    # ],
    # RegInstruction: [
    #     ('divq', 'div', 64),
    #     ('idivq', 'idiv', 64),
    #     ('imulq', 'imul', 64),
    #     ('mulq', 'mul', 64),
    #     ('negq', 'neg', 64),
    #     ('notq', 'not', 64),
    #     ('rolq', 'rol', 64),
    #     ('rorq', 'ror', 64),
    #     ('sarq', 'sar', 64),
    #     ('shlq', 'shl', 64),
    #     ('shrq', 'shr', 64),
    #     ('incrementq', 'inc', 64),
    #     ('decrementq', 'dec', 64)
    # ],
    # MemInstruction: [
    #     ('mulq', 'mul', 64),
    #     ('negq', 'neg', 64),
    #     ('sarq', 'sar', 64),
    #     ('shrq', 'shr', 64),
    #     ('incrementq', 'inc', 64),
    #     ('decrementq', 'dec', 64)
    # ],
    # RegMemImmInstruction: [
    #     ('imulq', 'imul', 64)
    # ],
    # RegRegImmInstruction: [
    #     ('imulq', 'imul', 64),
    #     ('shldq', 'shld', 64),
    #     ('shrdq', 'shrd', 64)
    # ],
    # RegImm32Instruction: [
    #     ('orq_imm32', 'or', 64),
    #     ('subq_imm32', 'sub', 64)
    # ],
    # Pop2Instruction: [
    #     ('pop2', 'pop2', 64),
    #     ('pop2p', 'pop2p', 64)
    # ],
    # Push2Instruction: [
    #     ('push2', 'push2', 64),
    #     ('push2p', 'push2p', 64)
    # ],
    # --- NDD instructions ---
    RegNddInstruction: [
        ('eidivq', 'idiv', 64, False),
        ('eidivq', 'idiv', 64, True),
        ('edivq', 'div', 64, False),
        ('edivq', 'div', 64, True),
        ('eimulq', 'imul', 64, False),
        ('eimulq', 'imul', 64, True),
        ('emulq', 'mul', 64, False),
        ('emulq', 'mul', 64, True),
    ],
    MemNddInstruction: [
        ('emulq', 'mul', 64, False),
        ('emulq', 'mul', 64, True),
    ],
    RegRegNddInstruction: [
        ('eimulq', 'imul', 64, False),
        ('eimulq', 'imul', 64, True),
        ('elzcntq', 'lzcnt', 64, False),
        ('elzcntq', 'lzcnt', 64, True),
        ('enegq', 'neg', 64, False),
        ('enegq', 'neg', 64, True),
        ('enotq', 'not', 64, None),
        ('epopcntq', 'popcnt', 64, False),
        ('epopcntq', 'popcnt', 64, True),
        ('erolq', 'rol', 64, False),
        ('erolq', 'rol', 64, True),
        ('erorq', 'ror', 64, False),
        ('erorq', 'ror', 64, True),
        ('esalq', 'sal', 64, False),
        ('esalq', 'sal', 64, True),
        ('esarq', 'sar', 64, False),
        ('esarq', 'sar', 64, True),
        # ('edecq', 'dec', 64, False), # protected
        # ('edecq', 'dec', 64, True), # protected
        # ('eincq', 'inc', 64, False), # protected
        # ('eincq', 'inc', 64, True), # protected
        ('eshlq', 'shl', 64, False),
        ('eshlq', 'shl', 64, True),
        ('eshrq', 'shr', 64, False),
        ('eshrq', 'shr', 64, True),
        ('etzcntq', 'tzcnt', 64, False),
        ('etzcntq', 'tzcnt', 64, True),
    ],
    RegMemNddInstruction: [
        # ('eimulq', 'imul', 64, False), # use vex instead of evex
        # ('eimulq', 'imul', 64, True), # use vex instead of evex
        ('elzcntq', 'lzcnt', 64, False),
        ('elzcntq', 'lzcnt', 64, True),
        ('enegq', 'neg', 64, False),
        ('enegq', 'neg', 64, True),
        ('epopcntq', 'popcnt', 64, False),
        ('epopcntq', 'popcnt', 64, True),
        ('esalq', 'sal', 64, False),
        ('esalq', 'sal', 64, True),
        ('esarq', 'sar', 64, False),
        ('esarq', 'sar', 64, True),
        # ('edecq', 'dec', 64, False), # protected
        # ('edecq', 'dec', 64, True), # protected
        # ('eincq', 'inc', 64, False), # protected
        # ('eincq', 'inc', 64, True), # protected
        ('eshrq', 'shr', 64, False),
        ('eshrq', 'shr', 64, True),
        ('etzcntq', 'tzcnt', 64, False),
        ('etzcntq', 'tzcnt', 64, True),
    ],
    RegMemRegNddInstruction: [
        ('eaddq', 'add', 64, False),
        ('eaddq', 'add', 64, True),
        ('eandq', 'and', 64, False),
        ('eandq', 'and', 64, True),
        ('eorq', 'or', 64, False),
        ('eorq', 'or', 64, True),
        ('esubq', 'sub', 64, False),
        ('esubq', 'sub', 64, True),
        ('exorq', 'xor', 64, False),
        ('exorq', 'xor', 64, True),
    ],
    RegMemImmNddInstruction: [
        ('eaddq', 'add', 64, False),
        ('eaddq', 'add', 64, True),
        ('eandq', 'and', 64, False),
        ('eandq', 'and', 64, True),
        ('eimulq', 'imul', 64, False),
        ('eimulq', 'imul', 64, True),
        ('eorq', 'or', 64, False),
        ('eorq', 'or', 64, True),
        ('esalq', 'sal', 64, False),
        ('esalq', 'sal', 64, True),
        ('esarq', 'sar', 64, False),
        ('esarq', 'sar', 64, True),
        ('eshrq', 'shr', 64, False),
        ('eshrq', 'shr', 64, True),
        ('esubq', 'sub', 64, False),
        ('esubq', 'sub', 64, True),
        ('exorq', 'xor', 64, False),
        ('exorq', 'xor', 64, True),
    ],
    RegRegImmNddInstruction: [
        ('eaddq', 'add', 64, False),
        ('eaddq', 'add', 64, True),
        ('eandq', 'and', 64, False),
        ('eandq', 'and', 64, True),
        # ('eimulq', 'imul', 64, False), # use vex instead of evex
        # ('eimulq', 'imul', 64, True), # use vex instead of evex
        ('eorq', 'or', 64, False),
        ('eorq', 'or', 64, True),
        ('erclq', 'rcl', 64, None),
        ('erolq', 'rol', 64, False),
        ('erolq', 'rol', 64, True),
        ('erorq', 'ror', 64, False),
        ('erorq', 'ror', 64, True),
        ('esalq', 'sal', 64, False),
        ('esalq', 'sal', 64, True),
        ('esarq', 'sar', 64, False),
        ('esarq', 'sar', 64, True),
        ('eshlq', 'shl', 64, False),
        ('eshlq', 'shl', 64, True),
        ('eshrq', 'shr', 64, False),
        ('eshrq', 'shr', 64, True),
        ('esubq', 'sub', 64, False),
        ('esubq', 'sub', 64, True),
        ('exorq', 'xor', 64, False),
        ('exorq', 'xor', 64, True),
    ],
    RegRegMemNddInstruction: [
        ('eaddq', 'add', 64, False),
        ('eaddq', 'add', 64, True),
        ('eandq', 'and', 64, False),
        ('eandq', 'and', 64, True),
        ('eorq', 'or', 64, False),
        ('eorq', 'or', 64, True),
        ('eimulq', 'imul', 64, False),
        ('eimulq', 'imul', 64, True),
        ('esubq', 'sub', 64, False),
        ('esubq', 'sub', 64, True),
        ('exorq', 'xor', 64, False),
        ('exorq', 'xor', 64, True),
    ],
    RegRegRegNddInstruction: [
        ('eaddq', 'add', 64, False),
        ('eaddq', 'add', 64, True),
        # ('eadcxq', 'adcx', 64, None), # reverse
        # ('eadoxq', 'adox', 64, None), # reverse
        ('eandq', 'and', 64, False),
        ('eandq', 'and', 64, True),
        # ('eimulq', 'imul', 64, False), # reverse
        # ('eimulq', 'imul', 64, True), # reverse
        ('eorq', 'or', 64, False),
        ('eorq', 'or', 64, True),
        ('esubq', 'sub', 64, False),
        ('esubq', 'sub', 64, True),
        ('exorq', 'xor', 64, False),
        ('exorq', 'xor', 64, True),
    ],
    RegRegRegImmNddInstruction: [
        # ('eshldq', 'shld', 64, False), # reverse
        # ('eshldq', 'shld', 64, True), # reverse
        # ('eshrdq', 'shrd', 64, False), # reverse
        # ('eshrdq', 'shrd', 64, True), # reverse
    ],
    CondRegRegRegNddInstruction: [
        ('ecmovq', 'cmov', 64, key) for key in cond_to_suffix.keys()
    ],
    CondRegRegMemNddInstruction: [
        ('ecmovq', 'cmov', 64, key) for key in cond_to_suffix.keys()
    ],
    RegRegImm32NddInstruction: [
        ('eorq_imm32', 'or', 64, False),
        ('eorq_imm32', 'or', 64, False),
        ('esubq_imm32', 'sub', 64, False), 
        ('esubq_imm32', 'sub', 64, True), 
    ],
}

for RegOp, ops in instruction_set.items():
    generate(RegOp, ops, True)

print("#ifdef _LP64")

for RegOp, ops in instruction_set64.items():
    generate(RegOp, ops, False)

print("#endif // _LP64")

outfile.close()

subprocess.check_call([X86_AS, "x86ops.s", "-o", "x86ops.o",])
subprocess.check_call([X86_OBJCOPY, "-O", "binary", "-j", ".text", "x86ops.o", "x86ops.bin"])

infile = open("x86ops.bin", "rb")
bytes = bytearray(infile.read())
infile.close()
disassembly_text = subprocess.check_output([OBJDUMP, "-M", "intel", "-d", "x86ops.o", "--insn-width=16"], text=True)
lines = disassembly_text.split("\n")
instruction_regex = re.compile(r'^\s*([0-9a-f]+):\s*([0-9a-f\s]+?)(?:\s{2,})')
instructions = []

for i, line in enumerate(lines):
    match = instruction_regex.match(line)
    if match:
        offset = int(match.group(1), 16)
        insns = match.group(2).split()
        binary_code = ", ".join([f"0x{insn}" for insn in insns])
        length = len(insns)
        instructions.append((length, binary_code))

print()
print("  static const uint8_t insns[] =")
print("  {")
print_with_ifdef(ifdef_flags, instructions, lambda x: x[1], items_per_line=1)
print("  };")
print("  static const unsigned int insns_lens[] =")
print("  {")
print_with_ifdef(ifdef_flags, instructions, lambda x: x[0], items_per_line=8)
print()
print("  };")
print()
print("  static const char* insns_strs[] =")
print("  {")
print_with_ifdef(ifdef_flags, instrs, lambda x: f"\"{x}\"", items_per_line=1)
print("  };")

print("// END  Generated code -- do not edit")

for f in ["x86ops.s", "x86ops.o", "x86ops.bin"]:
    os.remove(f)