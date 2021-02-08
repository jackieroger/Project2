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

def test_partitioning():
	assert True

def test_hierarchical():
	assert True
