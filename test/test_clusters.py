import pytest
import numpy as np
from clusters import algs

# Test I/O of ligand info
def test_ligand():
	# Info for zeroth ligand
	id0 = int(0)
	score0 = float(-1.3)
	on_bits0 = np.array([360, 489, 915])
	# Load ligands
	ligs = algs.load_ligands("ligand_information.csv")
	# Check that zeroth ligand info is correct
	lig0 = ligs[0]
	assert lig0.id == id0
	assert lig0.score == score0
	assert np.array_equal(lig0.on_bits, on_bits0)

def test_partitioning():
	assert True

def test_hierarchical():
	assert True
