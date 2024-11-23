library("adjclust")
library("matrixStats")
library("snpStats")


adjclust <- function(bed = 'data.bed', bim = 'data.bim', fam = 'data.fam') 
	{
	# Read plink data as  SNPmtarix object
	geno <- read.plink(bed, bim, fam)
	# Exctrat genotype
	genotype <- geno$genotypes
	p <- ncol(genotype)

	# Plot LD between SNPs
	ld_ <- snpStats::ld(genotype, stats = "R.squared", depth = p-1)
	png(file="LD_plot.png", width = 1000, height = 1000)
	image(ld_, lwd = 0)
	dev.off()

	# Adjacency-constrained Hierarchical Agglomerative Clustering

	fit <- snpClust(genotype, stats = "R.squared")
  	sel_clust <- select(fit)
  
	# Save LD-groups labels for further analysis 
	saveRDS(sel_clust, file = "LD_groups.rds")

	# Display rectangular dendrogram 
	png(file ="dendogram_rec.png",  width = 1000, height = 1000)
	plot(fit, type = "rectangle", leaflab = "perpendicular")
	dev.off()
	}