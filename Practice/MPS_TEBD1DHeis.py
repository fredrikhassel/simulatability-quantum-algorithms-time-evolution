import quimb as qu
import quimb.tensor as qtn
import numpy as np
import matplotlib.pyplot as plt

# |00..0 1 00..0 1 00..0> LENGTH 44
n = 44
zeros = '0' * ((n - 2) // 3)
binary = zeros + '1' + zeros + '1' + zeros

# Bitstring -> MPS
psi0 = qtn.MPS_computational_state(binary)
psi0.show()

# Using built-in Heisenberg Hamiltonian with h=0
H = qtn.ham_1d_heis(n)

# Setting up TEBD object
tebd = qtn.TEBD(psi0, H)
tebd.split_opts['cutoff'] = 1e-12

# Times
ts = np.linspace(0, 50, 100)

# Observables
mz_t_j = []  # z-magnetization
be_t_b = []  # block entropy
sg_t_b = []  # schmidt gap

# bonds and sites
js = np.arange(0, n)
bs = np.arange(1, n)

# Sampling TEBD at times
for psit in tebd.at_times(ts, tol=1e-3):
    # Local observables
    mz_j = []
    be_b = []
    sg_b = []

    # Orthogonality center = site where tensors are orthogonal
    info = {"cur_orthog": None}  # Current orthogonality center

    # Adding current datapoints using orthogonality center
    for j in range(1,n):
        mz_j.append(psit.magnetization(j, info=info))
        if j < n - 1:
            be_b.append(psit.entropy(j, info=info))
            sg_b.append(psit.schmidt_gap(j, info=info))
    mz_t_j.append(mz_j)
    be_t_b.append(be_b)
    sg_t_b.append(sg_b)

# Showing the resultant state and error
tebd.pt.show()
print("Rough upper error estimate: "+str(tebd.err))
print("Normalization of final state: "+str(tebd.pt.H @ tebd.pt))

# Checking conservation of energy
H = qtn.MPO_ham_heis(n)
print("Initial energy:", qtn.expec_TN_1D(psi0.H, H, psi0))
print("Final energy:", qtn.expec_TN_1D(tebd.pt.H , H, tebd.pt))

# Plotting data
with plt.style.context(qu.NEUTRAL_STYLE):
    plt.figure(figsize=(12, 7))

    # plot the magnetization
    ax1 = plt.subplot(131)
    plt.pcolormesh(np.real(mz_t_j), vmin=-0.5, vmax=0.5)
    plt.set_cmap('RdYlBu')
    plt.colorbar()
    plt.title('Z-Magnetization')
    plt.xlabel('Site')
    plt.ylabel('time [ $Jt$ ]')

    # plot the entropy
    ax2 = plt.subplot(132, sharey=ax1)
    plt.pcolormesh(be_t_b)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.set_cmap('viridis'), plt.colorbar()
    plt.title('Block Entropy')
    plt.xlabel('Bond')

    # plot the schmidt gap
    ax3 = plt.subplot(133, sharey=ax1)
    plt.pcolormesh(sg_t_b, vmin=0, vmax=1)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.set_cmap('magma_r')
    plt.colorbar()
    plt.title('Schmidt Gap')
    plt.xlabel('Bond')

    plt.show()