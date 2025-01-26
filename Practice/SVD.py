import numpy as np
import matplotlib.pyplot as plt

# import image
img = plt.imread("OGthumbnail_mps_simulation.png")

# only look at one color channel for demonstration purposes
img = img[:, :, 0]

# Perform SVD
U, Lambda, Vd = np.linalg.svd(img)

# Keep only the 50 largest singular values and vectors
chi = 50
U_compressed = U[:, :chi]
Lambda_compressed = Lambda[:chi]
Vd_compressed = Vd[:chi]

# Reconstruct the compressed image
compressed_img = U_compressed @ np.diag(Lambda_compressed) @ Vd_compressed

fig, axs = plt.subplots(ncols=2)
ax = axs[0]
ax.imshow(img, vmin=0, vmax=1)
ax.set_title("Uncompressed image")

ax = axs[1]
ax.imshow(compressed_img, vmin=0, vmax=1)
ax.set_title("Compressed image")

plt.show()

size_original = np.prod(img.shape)
size_compressed = np.prod(U_compressed.shape) + np.prod(Lambda_compressed.shape) + np.prod(Vd_compressed.shape)

print(f"original image size: {size_original}, compressed image size: {size_compressed}, factor {size_original/size_compressed:.3f} saving")

_, Lambda, _ = np.linalg.svd(img) # recompute the full spectrum
plt.plot(Lambda)
plt.xlabel("index $i$")
plt.ylabel("$\\Lambda_i$")
plt.show()