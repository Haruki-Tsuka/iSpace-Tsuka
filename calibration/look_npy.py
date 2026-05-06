import numpy as np

mtx = np.load("results/mtx.npy")
dist = np.load("results/dist.npy")

print("===== mtx =====")
print("shape:", mtx.shape)
print("dtype:", mtx.dtype)
print(mtx)

print("\n===== dist =====")
print("shape:", dist.shape)
print("dtype:", dist.dtype)
print(dist)