# **MuGLasso and SMuGLasso Stability Analysis**

This repository contains an end-to-end pipeline for applying **Multi-task Group Lasso (MuGLasso)** and **Sparse Multi-task Group Lasso (SMuGLasso)** to analyze stability in GWAS datasets. It includes scripts for preprocessing genotype data, clustering SNPs into LD groups, and evaluating feature stability using bootstrap resampling.



---

## **Overview**
This repository is an implementation of MuGLasso and SMuGLasso applied to GWAS
This project utilizes techniques inspired by the **Gap Safe Screening** package, which can be found [here](https://github.com/EugeneNdiaye/Gap_Safe_Rules).
We extend its usage to multi-task group lasso, sparse multi-task group lasso in logistic loss problems.


---

## **Features**
- Preprocess genotype datasets into a format suitable for SMuGLasso.
- Cluster SNPs into LD groups using R-based **adjclust** library.
- Merge LD groups across populations.
- Perform stability selection on MuGLasso or/and SMuGLasso.
- Evaluate the stability of the selection.
---

## **Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/asmanouira/SMuGLasso
cd SMuGLasso
```
## **Usage**

### **Input Data**

* Genotype data in `.raw` plink format with the first 6 columns for metadata and SNP columns for genotype values (0, 1, or 2).
* Diverse GWAS data can be simulated using GWAsimulator by following the tutorial in the github repository: [GWAS-admixed-population-simulator](https://github.com/asmanouira/GWAS-admixed-population-simulator).

### **Running the Pipeline**

Execute the pipeline by running:

```
bash
Copy code
python main.py
```

***

## **Scripts Description**

### **1. `main.py`**

The main script that connects all parts of the pipeline:

* Reads genotype data and transforms it for MuGLasso.
* Calls the R script for LD group clustering.
* Merges LD groups across populations.
* Runs MuGLasso and SMuGLasso with stability selection.

### **2. `input_generator.py`**

Contains:

* `input_matrix`: Converts genotype data into the MuGLasso-compatible input format.
* `plot_data`: Visualizes the design matrix.

### **3. `LD_groups_clustering.R`**

Uses R libraries to:

* Compute LD between SNPs.
* Perform hierarchical clustering of SNPs into LD groups.
* Save the resulting LD groups as `LD_groups.rds`.

### **4. `merge_shared_groups.py`**

Merges LD group identifiers across populations into a unified set of shared group indices.

### **5. `muglasso.py` and `smuglasso.py`**

Python implementations of MuGLasso and SMuGLasso models, leveraging optimized Cython routines.

### **6. `multitask_stability_selection.py`**

Performs bootstrap-based stability selection to compute stability scores for MuGLasso and SMuGLasso.

***

## **Dependencies**

### **Python Libraries**

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `Cython`

### **R Packages**

* `adjclust`
* `matrixStats`
* `snpStats`

***

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for details.

***

