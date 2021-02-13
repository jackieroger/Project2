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
	Loads ligand information from csv file and creates an array of Ligands.

	Parameters
	----------
	ligand_file : str
		Name of csv file containing information for each ligand.

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

def _calc_jaccard_distance(p1, p2):
	"""
    Private utility function that calculates jaccard distance between 2 points. This is useful for both clustering algorithms.

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

class HierarchicalClustering():
	"""
	Implements hierarchical clustering for an array of coordinates.

	Attributes
	----------
	num_clusters : int
		Number of clusters.
	clusters : np.array
		Clusters generated from doing hierarchical clustering.

	Parameters
	----------
	num_clusters : int
		Number of clusters. Saved as a class attribute.
	"""

	def __init__(self, num_clusters):
		"""
		Initializes HierarchicalClustering object, and creates clusters.
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
			c[i].members = np.asarray([i])
		return c

	def _init_distance_matrix(self, points):
		"""
		Private helper method for cluster(). Initializes pairwise distance matrix for points. Only values for the upper righthand triangle are used for clustering.

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
		dm = np.full(shape=(len(points), len(points)), fill_value=np.inf)
		# Fill in dm
		# Go through each row
		for i in range(len(points)):
			# Go through each column
			for j in range(i+1, len(points)):
				# Calculate distance for each cell in upper right triangle of dm (excluding diagonal)
				p1 = points[i,]
				p2 = points[j,]
				dm[i, j] = _calc_jaccard_distance(p1, p2)
		return dm

	def _do_hc(self, c, dm):
		"""
		Private helper method for cluster(). Do hierarchical clustering to form the specified number of clusters.

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
		# Set the current number of clusters
		curr_num_clusters = len(c)
		# Do clustering
		while curr_num_clusters > self.num_clusters:
			# Find location of min value in dm
			min_dist_loc = np.unravel_index(np.argmin(dm, axis=None), dm.shape)
			i = min_dist_loc[0]
			j = min_dist_loc[1]
			# Combine clusters by pulling out members of each, deleting original cluster,
			# and appending a new one with members from both
			mem_i = c[i].members
			mem_j = c[j].members
			c = np.delete(c, [i, j])
			c = np.append(c, Cluster())
			c[-1].members = np.concatenate((mem_i, mem_j))
			# Update dm
			# Add a new col
			dm = np.append(dm, np.ones((1, len(dm))), axis=0)
			dm = np.append(dm, np.ones((len(dm), 1)), axis=1)
			# Update new column (using single linkage)
			for x in range(len(dm)-1):
				options = [dm[x,i], dm[x,j], dm[i,x], dm[j,x]]
				dm[x, len(dm)-1] = min(options)
			# Delete rows & cols for 2 old clusters
			dm = np.delete(dm, [i, j], axis=0)
			dm = np.delete(dm, [i, j], axis=1)
			# Update num clusters
			curr_num_clusters -= 1
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

class PartitionClustering():
	pass

def get_cluster_assignments(self, c):
	"""
	Gets the cluster assignments for points that have already been clustered.

	Parameters
	----------
	c : np.array
		Array of Clusters.

	Returns
	-------
	cl_nums : dict
		Cluster assignments for each point (each key is a point and each value is its cluster assignment).
	"""
	cl_nums = {}
	for i in range(len(c)):
		for m in c[i].members:
			cl_nums[m] = i
	return cl_nums
