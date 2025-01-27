import numpy as np

# All credit: https://pennylane.ai/qml/demos/tutorial_mps#area-law

# psi_sigma1,sigma2,sigma3
n = 3 # three sites = three legs
psi = np.random.rand(2**3)
psi = psi / np.linalg.norm(psi)  # random, normalized state vector
psi = np.reshape(psi, (2, 2, 2)) # rewrite psi as rank-n tensor

# reshape vector to matrix
psi = np.reshape(psi, (2, 2**(n-1)))
# SVD to split off first site
U, Lambda, Vd = np.linalg.svd(psi, full_matrices=False)

# U tensors are shaped (virtual_left, physical, virtual_right). 
# A dummy index of size 1 is added to the first site for consistent tensor shapes in the MPS.
Us = []
U = np.reshape(U, (1, 2, 2)) # mu1, s2, mu2
Us.append(U)

# This procedure is repeated for all sites. The first step is unique as Uσ₁μ₁ is a vector for each σ₁. 
# When splitting ψ′μ₁,(σ₂σ₃), the virtual bond is combined with the current site, 
# while remaining sites form the other leg of the SVD matrix.
psi_remainder = np.diag(Lambda) @ Vd                 # mu1 (s2 s3)
psi_remainder = np.reshape(psi_remainder, (2*2, 2))  # (mu1 s2), s3
U, Lambda, Vd = np.linalg.svd(psi_remainder, full_matrices=False)

U = np.reshape(U, (2, 2, 2)) # mu1, s2, mu2
Us.append(U)

# Confirming our shapes
print("U's shape: "+str(U.shape))
print("Λ's shape: "+str(Lambda.shape))
print("V†'s shape: "+str(Vd.shape))

# Redefining the remainder for the last site
psi_remainder = np.diag(Lambda) @ Vd                 # mu1 (s2 s3)
psi_remainder = np.reshape(psi_remainder, (2*2, 1))  # (mu1 s2), s3
U, Lambda, Vd = np.linalg.svd(psi_remainder, full_matrices=False)

U = np.reshape(U, (2, 2, 1)) # mu1, s2, mu2
Us.append(U)

# Confirming our redefined shapes
print("U's shape: "+str(U.shape))
print("Λ's shape: "+str(Lambda.shape))
print("V†'s shape: "+str(Vd.shape))

# Reconstructing the original state:
print(f"Shapes of Us: {[_.shape for _ in Us]}")

psi_reconstruct = Us[0]
for i in range(1, len(Us)):
    # contract the rightmost with the left most index
    psi_reconstruct = np.tensordot(psi_reconstruct, Us[i], axes=1)

print(f"Shape of reconstructed psi: {psi_reconstruct.shape}")
# remove dummy dimensions
psi_reconstruct = np.reshape(psi_reconstruct, (2, 2, 2))
# original shape of original psi
psi = np.reshape(psi, (2, 2, 2))

print("Reconstructed state matching initial: "+str(np.allclose(psi, psi_reconstruct))+"\n")

def split(M, bond_dim):
    """Split a matrix M via SVD and keep only the ``bond_dim`` largest entries."""
    U, S, Vd = np.linalg.svd(M, full_matrices=False)
    bonds = len(S)
    Vd = Vd.reshape(bonds, 2, -1)
    U = U.reshape((-1, 2, bonds))

    # keep only chi bonds
    chi = np.min([bonds, bond_dim])
    U, S, Vd = U[:, :, :chi], S[:chi], Vd[:chi]
    return U, S, Vd

def dense_to_mps(psi, bond_dim):
    """Turn a state vector ``psi`` into an MPS with bond dimension ``bond_dim``."""
    Ms = []
    Ss = []

    psi = np.reshape(psi, (2, -1))   # split psi[2, 2, 2, 2..] = psi[2, (2x2x2...)]
    U, S, Vd = split(psi, bond_dim)  # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]

    Ms.append(U)
    Ss.append(S)
    bondL = Vd.shape[0]
    psi = np.tensordot(np.diag(S), Vd, 1)

    for _ in range(n-2):
        psi = np.reshape(psi, (2*bondL, -1)) # reshape psi[2 * bondL, (2x2x2...)]
        U, S, Vd = split(psi, bond_dim) # psi[2, (2x2x..)] = U[2, mu] S[mu] Vd[mu, (2x2x2x..)]
        Ms.append(U)
        Ss.append(S)

        psi = np.tensordot(np.diag(S), Vd, 1)
        bondL = Vd.shape[0]

    # dummy step on last site
    psi = np.reshape(psi, (-1, 1))
    U, _, _ = np.linalg.svd(psi, full_matrices=False)

    U = np.reshape(U, (-1, 2, 1))
    Ms.append(U)

    return Ms, Ss

# Showing the different bond size growth
n = 12
bond_dim = 10000
psi = np.random.rand(*[2]*n)
psi = psi/np.linalg.norm(psi)
Ms, Ss = dense_to_mps(psi, bond_dim)
[print(M.shape) for M in Ms]
print("\n")

n = 12
bond_dim = 5
psi = np.random.rand(*[2]*n)
psi = psi/np.linalg.norm(psi)
Ms, Ss = dense_to_mps(psi, bond_dim)
[print(M.shape) for M in Ms]
print("\n")

# Checking the left orthogonality of our U's:
for i in range(len(Ms)):
    id_ = np.tensordot(Ms[i].conj(), Ms[i], axes=([0, 1], [0, 1]))
    is_id = np.allclose(id_, np.eye(len(id_)))
    print(f"U[{i}] is left-orthonormal: {is_id}")