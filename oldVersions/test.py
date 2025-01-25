from pyquest import Register, Circuit
import numpy as np

# Register of 10 cubits
reg = Register(2)

# 2 ways of creating identical copy
copied_reg = Register(copy_reg=reg)
copied_reg = reg.copy()

# copy over the amplitudes
copied_reg.copy_from(reg)

# Same properties but not copy over the state
like_reg = Register.zero_like(reg)

# Completely clear a register
reg.init_blank_state()

# Register manipulation
reg /= (np.sqrt(reg.total_prob)+0.1) # total prob gives total probability of a state

reg1 = Register(10)
reg2 = Register(10)

newReg = reg1 + reg2 - reg2

reg1 += reg2
reg2 -= reg1 # register from the other in-place.

c = 1 + 2j
newReg = reg * c / c # a (complex) scalar c and creating a new register.

reg *= c 
reg/= c # a (complex) scalar in-place.

