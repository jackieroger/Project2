import numpy as np

class Ligand():
	"""
	Parameters
	----------
	id : int
		Identification number. Saved as class attribute.
	score : float
		AutoDock Vina score. Saved as class attribute.
	on_bits : np.array
		The “On” bits in the Extended Connectivity Fingerprint. Saved as class attribute.
	"""

	def __init__(self, id, score, on_bits):
		self.id = int(id)
		self.score = float(score)
		self.on_bits = np.array(on_bits)


def load_ligands(ligand_file):
	"""
	Parameters
	----------
	ligand_file : str
		Name of file containing information for each ligand.

	Returns
	-------
	ligands : list
		List of Ligand objects.
	"""
	# Open up the ligand info file and extract info
	with open(ligand_file) as lf:
		lines = [line.strip() for line in lf if line[0].isdigit()]
	# Make list of ligands
	ligands = []
	# Split by quotes and commas to get the ligand info from each line
	for l in lines:
		ls = l.split("\"")
		lsc = ls[0].split(",")
		id = lsc[0]
		score = lsc[1]
		on_bits = np.array([int(ob) for ob in ls[1].split(",")])
		ligands.append(Ligand(id, score, on_bits))
	return ligands

class Clustering():
	pass

class HierarchicalClustering():
	pass

class PartitionClustering():
	pass