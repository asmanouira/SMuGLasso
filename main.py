import numpy as np
import pandas as pd
from muglasso import MuGLasso
from smuglasso import SMuGLasso
from input_generator import input_matrix, plot_data
from merge_shared_groups import merge_multiple_LD_groups
from multitask_stability_selection import stability_selection

def main():



    # Parameters for the simulated dataset
    n_samples = 100  # Number of individuals
    n_snps = 1000  # Number of SNPs (features)
    n_tasks = 3  # Number of tasks (if required)

    # Step1: Simulate the first 6 non-genotype columns as in .raw files
    columns_non_genotype = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    non_genotype_data = {
        'FID': [f"F{i+1}" for i in range(n_samples)],
        'IID': [f"I{i+1}" for i in range(n_samples)],
        'PAT': [0] * n_samples,
        'MAT': [0] * n_samples,
        'SEX': np.random.choice([1, 2], n_samples),  # Randomly assign sex (1 = male, 2 = female)
        'PHENOTYPE': np.random.choice([1, 2], n_samples),  # Randomly assign binary phenotype
    }

    # Simulate SNP data (values 0, 1, 2)
    genotype_data = np.random.choice([0, 1, 2], size=(n_samples, n_snps))

    # Combine non-genotype and genotype data
    genotype_df = pd.DataFrame(non_genotype_data)
    snp_columns = [f"SNP{i+1}" for i in range(n_snps)]
    genotype_snp_df = pd.DataFrame(genotype_data, columns=snp_columns)
    genotype_full_df = pd.concat([genotype_df, genotype_snp_df], axis=1)

    # Save as a .raw file (space-delimited text)
    genotype_full_df.to_csv("genotype_data.raw", sep=" ", index=False)
    print("Simulated genotype data saved to 'genotype_data.raw'.")

    # Step 1: Data Transformation
    print("Step 2: Data Transformation...")
    data_path = "genotype_data.raw"  # Replace with your genotype file path
    n_tasks = 3  # Number of tasks
    n_samples = 100  # Total number of samples
    n_samples_per_task = [30, 30, 40]  # Samples per task
    n_features = 1000  # Number of features (SNPs)

    # Transform data to MuGLasso input format
    X_mt = input_matrix(data_path, n_tasks, n_samples, n_samples_per_task, n_features)
    plot_data(X_mt)  # Save design matrix visualization
    print(f"Transformed data shape: {X_mt.shape}")

    # Step 2: LD Groups Clustering (This step is done in R)
    print("Step 3: Perform LD Groups Clustering in R...")
    # Run the LD_groups_clustering.R script externally to generate the LD groups.
    # The output should be an RDS file with the LD groups. Ensure it's saved as "LD_groups.rds".

    # Step 3: Merge LD Groups
    print("Step 4: Merging LD Groups...")
    ld_groups_file = "LD_groups.rds"  # Output from the R script
    ld_groups = pd.read_rds(ld_groups_file)  # Replace with a Python RDS reader or pre-convert it
    merged_ld_groups = merge_multiple_LD_groups(ld_groups)

    print(f"Merged LD Groups: {merged_ld_groups}")

    # Step 4: Run MuGLasso and SMuGLasso with Stability Selection
    print("Step 5: Running Stability Selection for MuGLasso and SMuGLasso...")
    y = np.random.rand(n_samples)  # Example target data, replace with actual data
    lambdas = np.logspace(-3, 0, 5)  # Regularization parameter values
    n_bootstrap_iterations = 50
    bootstrap_sets = [np.random.choice(range(n_samples), size=n_samples // 2, replace=False)
                      for _ in range(5)]  # Example bootstrap sets
    seed = 42

    # Initialize MuGLasso and SMuGLasso models
    muglasso_model = MuGLasso(loss='quadratic', tau=0.5, lambda2=0.1, verbose=True)
    smuglasso_model = SMuGLasso(loss='quadratic', alpha=0.5, lambda1=0.5, lambda2=0.1, verbose=True)

    # Perform stability selection for MuGLasso
    print("Running MuGLasso...")
    muglasso_scores = stability_selection(muglasso_model, lambdas, n_bootstrap_iterations, X_mt, y, bootstrap_sets, seed)
    print(f"MuGLasso Stability Scores:\n{muglasso_scores}")

    # Perform stability selection for SMuGLasso
    print("Running SMuGLasso...")
    smuglasso_scores = stability_selection(smuglasso_model, lambdas, n_bootstrap_iterations, X_mt, y, bootstrap_sets, seed)
    print(f"SMuGLasso Stability Scores:\n{smuglasso_scores}")

    # Save Results
    print("Saving results...")
    np.savetxt("muglasso_stability_scores.csv", muglasso_scores, delimiter=",")
    np.savetxt("smuglasso_stability_scores.csv", smuglasso_scores, delimiter=",")

    print("Pipeline complete!")

if __name__ == "__main__":
    main()
