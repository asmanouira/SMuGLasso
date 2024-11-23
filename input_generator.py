import numpy as np
from matplotlib import pyplot as plt
import pandas as pd  


def input_matrix(data, n_tasks, n_samples, n_samples_per_task, n_features):

	"""
	Design input matrix for the Multi-task group Lasso algorithm (MuGLAsso)


	Args:

		data : input matrix, here is the genotype array.
		n_tasks: integer: number of tasks
		n_samples: integer: number of total inputs
		n_samples_per_task: list of length n_tasks: number of samples per task
		n_features: number of features, here is the number of SNPs
		
	Returns:

		X_mt : the input matrix in MuGLasso format
	"""

	data = pd.read_csv(data, delim_whitespace=True)

	# Remove useless columns, 
	# Please comment the following line if your data is not plink recoded data (.raw file) with first 6 columns are non-genotype features
	X = data.iloc[:, 6:].values
	print(X.shape)
	X_mt = np.zeros((n_samples, n_features*n_tasks), dtype=np.int32)
	start_idx = 0
	end_idx = 0 
	
	for r in range(n_tasks):
	    end_idx += n_samples_per_task[r]
	    
	    X_mt[start_idx:end_idx, r*n_features:(r+1)*n_features] = X[start_idx:end_idx, :]
	    
	    start_idx += n_samples_per_task[r]

	return(X_mt)

def plot_data(X_mt):

	plt.imshow(X_mt, cmap='viridis', aspect='equal')
	plt.colorbar()
	plt.savefig("design_mat.png")