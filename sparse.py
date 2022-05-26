import numpy as np

matrix = np.load("compact_data/matrix.npy")
matrix = np.transpose(matrix)
print("calculating correlations...")
corrcoef = np.corrcoef(matrix)

print("selecting...")
matrix = matrix[np.any(np.abs(corrcoef) > 0.8, axis=1)]
print(matrix.shape)
