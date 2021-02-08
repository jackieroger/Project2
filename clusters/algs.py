import numpy as np

class Ligand():
	"""
	Represents a ligand and stores relevant information yielded from docking.

	Attributes
	----------
	id : int
		Identification number.
	score : float
		AutoDock Vina score.
	smiles : str
		SMILES string.
	on_bits : np.array
		The “on” bits in the Extended Connectivity Fingerprint.
	bit_string : np.array
		The 1024-length bit string of whether or not each bit is "on".

	Parameters
	----------
	id : int
		Identification number. Saved as class attribute.
	score : float
		AutoDock Vina score. Saved as class attribute.
	smiles : str
		SMILES string. Saved as class attribute.
	on_bits : np.array
		The “On” bits in the Extended Connectivity Fingerprint. Saved as class attribute.
	"""

	def __init__(self, id, score, smiles, on_bits):
		"""
		Initializes Ligand object, and creates dense representation of on bits.
		"""
		self.id = int(id)
		self.score = float(score)
		self.smiles = str(smiles)
		self.on_bits = np.array(on_bits)
		# Create bit string
		self.bit_string = np.zeros(1024, dtype=int)
		for i in self.on_bits:
			self.bit_string[i] = 1

def load_ligands(ligand_file):
	"""
	Makes an array of Ligands from a csv file containing information about ligands.

	Parameters
	----------
	ligand_file : str
		Name of file containing information for each ligand.

	Returns
	-------
	ligands : np.array
		Ligands.
	"""
	# Open up the ligand info file and extract info
	with open(ligand_file) as lf:
		lines = [line.strip() for line in lf if line[0].isdigit()]
	# Make list of ligands
	ligands = []
	# Get the ligand info from each line
	for l in lines:
		# Split by quotes
		ls = l.split("\"")
		# Split by commas
		lsc = ls[0].split(",")
		# Get id (before 0th comma)
		id = int(lsc[0])
		# Get score (between 0th & 1st comma)
		score = float(lsc[1])
		# Get smiles (between 1st & 2nd comma)
		smiles = str(lsc[2])
		# Get on bits (after 0th quote)
		on_bits = np.array([int(ob) for ob in ls[1].split(",")])
		ligands.append(Ligand(id, score, smiles, on_bits))
	# Convert to np array
	ligands = np.asarray(ligands)
	return ligands

class Cluster():
	"""
	Represents a cluster by containing an array of ids of the members of the cluster.

	Attributes
	----------
	members : np.array
		Unique ids associated with each member of the cluster.
	centroid : np.array
		Cluster centroid. Only created for partition clustering.

	Parameters
	----------
	p : bool
		Flag for whether or not you are using a partitioning algorithm. Default is False.
	"""

	def __init__(self, p=False):
		"""
		Initializes Cluster object, and creates the members array and centroid array (if applicable).
		"""
		self.members = np.array([], dtype=int)
		# If doing partition clustering, create the cluster centroid
		if p == True:
			self.centroid = np.array([], dtype=float)

class HierarchicalClustering():
	"""
	Implements hierarchical clustering for an array of coordinates.

	Attributes
	----------
	num_clusters : int
		Number of clusters.
	clusters : np.array
		Clusters generated from doing hierarchical clustering.
	"""

	def __init__(self, num_clusters):
		"""
		Initializes HierarchicalClustering object, and creates clusters.

		Parameters
		----------
		num_clusters : int
			Number of clusters. Saved as a class attribute.
		"""
		self.num_clusters = num_clusters
		self.clusters = np.array([])

	def _init_clusters(self, points):
		"""
		Private helper method for cluster(). Initializes clusters.

		Parameters
		----------
		points : np.array
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		c : np.array
			Clusters that are initialized such that each point is in its own Cluster.
		"""
		# Create c
		c = np.array([Cluster() for i in range(len(points))])
		# Add the cluster members to each cluster in c
		for i in range(len(points)):
			c[i].members = i
		return c

	def _calc_jaccard_distance(self, p1, p2):
		"""
		Private helper method for _init_distance_matrix(). Calculates jaccard distance between 2 points.

		Parameters
		----------
		p1 : np.array
			A point (represented as an array of values for each coordinate space).
		p2 : np.array
			Another point (also represented as an array of values for each dimension/feature).

		Returns
		-------
		d : float
			Distance between 2 points calculated using jaccard distance.
		"""
		# Jaccard distance is 1-(intersection/union)
		# Credit: Brenda Miao told me about the numpy bitwise_or and bitwise_and functions
		# which are useful for fast calculations of jaccard distance
		i = (p1 & p2).sum()
		u = (p1 | p2).sum()
		d = 1 - (i/u)
		return d

	def _init_distance_matrix(self, points):
		"""
		Private helper method for cluster(). Initializes pairwise distance matrix for points.

		Parameters
		----------
		points : np.array
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		dm : np.array
			Distance matrix that has the pairwise distances between points calculated using
			jaccard distance.
		"""
		# Create dm
		dm = np.ones(shape=(len(points), len(points)))
		# Fill in dm
		# Go through each row
		for i in range(len(points)):
			# Go through each column
			for j in range(i+1, len(points)):
				# Calculate distance for each cell in dm
				p1 = points[i,]
				p2 = points[j,]
				dm[i, j] = self._calc_jaccard_distance(p1, p2)
		return dm

	def _do_hc(self, c, dm):
		"""
		Private helper method for cluster(). Do hierarchical clustering to form the specified
		number of clusters.

		Parameters
		----------
		c : np.array
			Initialized Clusters (with each point having its own Cluster).
		dm : np.array
			Initialized distance matrix to use for clustering.

		Returns
		-------
		c : np.array
			Finalized Clusters from hierarchical clustering.
		"""

		return c

	def cluster(self, points):
		"""
		Clusters a set of points using hierarchical clustering.

		Parameters
		----------
		points : np.array
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		c : np.array
			Clusters yielded from hierarchical clustering of points.
		"""
		# Initialize clusters
		c = self._init_clusters(points)
		# Initialize distance matrix
		dm = self._init_distance_matrix(points)
		# Do hierarchical clustering
		c = self._do_hc(c, dm)
		self.clusters = c
		return c

ligs = load_ligands("ligand_information.csv")
lig_subset = ligs[0:5]
lig_coords = np.array([l.bit_string for l in lig_subset])
hc = HierarchicalClustering(2)
clusters = hc.cluster(lig_coords)

class PartitionClustering():
	pass