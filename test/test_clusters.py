import pytest
import numpy as np
from clusters import algs

# Test I/O of ligand info for 2 ligands
def test_ligand():
	# Info for first ligand
	id1 = int(1)
	score1 = float(-1.7)
	smiles1 = "O=N"
	on_bits1 = np.array([53, 623, 650])
	# Info for second ligand
	id2 = int(2)
	score2 = float(-2.7)
	smiles2 = "OC#N"
	on_bits2 = np.array([260, 291, 360, 674, 790, 807])
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	# Check that first ligand info is correct
	lig1 = ligs[0]
	assert lig1.id == id1
	assert lig1.score == score1
	assert lig1.smiles == smiles1
	assert np.array_equal(lig1.on_bits, on_bits1)
	# Check that second ligand info is correct
	lig2 = ligs[1]
	assert lig2.id == id2
	assert lig2.score == score2
	assert lig2.smiles == smiles2
	assert np.array_equal(lig2.on_bits, on_bits2)
	# Check that there are the right number of on bits in bit strings
	assert np.count_nonzero(lig1.bit_string) == 3
	assert np.count_nonzero(lig2.bit_string) == 6
	# As an extra check, make sure that the correct bits are 1 for lig1
	assert lig1.bit_string[53] == 1
	assert lig1.bit_string[623] == 1
	assert lig1.bit_string[650] == 1

# Test hierarchical clustering
def test_hierarchical():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	# Do clustering on subset 1
	lig_subset1 = ligs[0:10]
	lig_coords1 = np.array([l.bit_string for l in lig_subset1])
	hc1 = algs.HierarchicalClustering(2)
	c1 = hc1.cluster(lig_coords1)
	# Do clustering on subset 2
	lig_subset2 = ligs[10:20]
	lig_coords2 = np.array([l.bit_string for l in lig_subset2])
	hc2 = algs.HierarchicalClustering(3)
	c2 = hc2.cluster(lig_coords2)
	# Set true values

# Test partition clustering
def test_partitioning():
	assert True
