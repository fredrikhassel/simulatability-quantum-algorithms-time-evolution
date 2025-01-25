from pyquest import Register, Circuit, unitaries, gates

zerozero = 0
oneone   = 0
zeroone  = 0
onezero  = 0
N        = 10000

for i in range(N):

    # \00>
    reg = Register(2)

    # \+> ⊗ \0>
    # HADAMARD GATE
    H = unitaries.H(0) # transforms first qubit into + state

    # 1/√2 \00> + \11>
    # CNOT GATE
    X = unitaries.X(1, controls = 0) # Performs CNOT with first as control, second as target

    # Measuring both cubits
    # MEASUREMENT GATE
    M = gates.M([0,1])

    # Making circuit
    cir = Circuit([H,X,M])

    # Applying circuit
    measurement = reg.apply_circuit(cir)

    match measurement[0]:
        case [0,0]:
            zerozero += 1
        case [1,1]:
            oneone   += 1
        case [1,0]:
            onezero  += 1
        case [0,1]:
            zeroone  += 1
    
print("Running circuit "+str(N)+" times.")
print("Frequency of |00>: "+str(zerozero))
print("Frequency of |01>: "+str(zeroone))
print("Frequency of |10>: "+str(onezero))
print("Frequency of |11>: "+str(oneone))