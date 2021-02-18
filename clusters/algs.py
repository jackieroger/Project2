import numpy as np

### LIGANDS ###

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
	on_bits : np.array of ints
		The “on” bits in the Extended Connectivity Fingerprint.
	bit_string : np.array of ints
		The 1024-length bit string of whether or not each bit is "on".

	Parameters
	----------
	id : int
		Identification number. Saved as class attribute.
	score : float
		AutoDock Vina score. Saved as class attribute.
	smiles : str
		SMILES string. Saved as class attribute.
	on_bits : np.array of ints
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
	ligands : np.array of Ligands
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

### CLUSTERING ###

class Cluster():
	"""
	Represents a cluster by containing an array of ids of the members of the cluster.

	Attributes
	----------
	members : np.array of ints
		Unique ids associated with each member of the cluster.
	"""

	def __init__(self):
		"""
		Initializes Cluster object, and creates the members array.
		"""
		self.members = np.array([], dtype=int)

class HierarchicalClustering():
	"""
	Implements hierarchical clustering for an array of coordinates.

	Attributes
	----------
	num_clusters : int
		Number of clusters.
	clusters : np.array of Clusters
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
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		c : np.array of Clusters
			Clusters that are initialized such that each point is in its own Cluster.
		"""
		# Create c
		c = np.array([Cluster() for i in range(len(points))])
		# Add the cluster members to each cluster in c
		for i in range(len(points)):
			c[i].members = np.asarray([i])
		return c

	def _calc_jaccard_distance(self, p1, p2):
		"""
		Private helper method for _init_distance_matrix() that calculates jaccard distance between 2 points.

		Parameters
		----------
		p1 : np.array of ints
			A point (represented as an array of values for each dimension/feature).
		p2 : np.array of ints
			Another point (also represented as an array of values for each dimension/feature).

		Returns
		-------
		d : float
			Distance between 2 points calculated using jaccard distance.
		"""
		# Jaccard distance is 1-(intersection/union)
		i = (p1 & p2).sum()
		u = (p1 | p2).sum()
		d = 1 - (i / u)
		return d

	def _init_distance_matrix(self, points):
		"""
		Private helper method for cluster(). Initializes pairwise distance matrix for points. Only values for the upper righthand triangle are used for clustering.

		Parameters
		----------
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		dm : np.array of floats
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
				dm[i, j] = self._calc_jaccard_distance(p1, p2)
		return dm

	def _do_hc(self, c, dm):
		"""
		Private helper method for cluster(). Do hierarchical clustering to form the specified number of clusters.

		Parameters
		----------
		c : np.array of Clusters
			Initialized Clusters (with each point having its own Cluster).
		dm : np.array of floats
			Initialized distance matrix to use for clustering.

		Returns
		-------
		c : np.array of Clusters
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
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		c : np.array of Clusters
			Clusters yielded from hierarchical clustering of points.
		"""
		# Initialize clusters
		c = self._init_clusters(points)
		# Initialize distance matrix
		dm = self._init_distance_matrix(points)
		# Do hierarchical clustering
		c = self._do_hc(c, dm)
		# Save clusters as a class attribute
		self.clusters = c
		# Also return clusters
		return c

class Centroid():
	"""
	Represents a cluster centroid by containing an array of the coordinates for the centroid of the cluster. Used for partition clustering.

	Attributes
	----------
	coords : np.array of ints
		Coordinates of cluster centroid.
	"""

	def __init__(self):
		"""
		Initializes Centroid object, and creates the coordinates array.
		"""
		self.coords = np.array([], dtype=int)

class PartitionClustering():
	"""
	Implements partition clustering for an array of coordinates.

	Attributes
	----------
	num_clusters : int
		Number of clusters.
	max_iterations : int
		Maximum number of iterations.
	clusters : np.array of Clusters
		Clusters generated from doing hierarchical clustering.

	Parameters
	----------
	num_clusters : int
		Number of clusters. Saved as a class attribute.
	max_iterations : int
		Maximum number of iterations. Saved as a class attribute.
	"""

	def __init__(self, num_clusters, max_iterations):
		"""
		Initializes PartitionClustering object, and creates clusters.
		"""
		self.num_clusters = num_clusters
		self.max_iterations = max_iterations
		self.clusters = np.array([])

	def _calc_euclidean_distance(self, p1, p2):
		"""
		Private helper method for _init_centroids() and _do_pc(). Calculates the euclidean distance between two points.

		Parameters
		----------
		p1 : np.array of ints
			A point (represented as an array of values for each dimension/feature).
		p2 : np.array of ints
			Another point (also represented as an array of values for each dimension/feature).

		Returns
		-------
		d : float
			Distance between 2 points calculated using euclidean distance.
		"""
		d = np.sqrt(np.sum(np.square(p2 - p1)))
		return d

	def _init_centroids(self, points):
		"""
		Initializes centroids using kmeans++ initialization.

		Parameters
		----------
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		centroids : np.array of Centroids
			Centroids yielded from kmeans++ initialization.
		"""
		# Randomly select first centroid from points
		centroids = points[np.random.randint(len(points))]
		# Make an array representing the distance from each point to each centroid
		# where the rows are points and the columns are centroids. Fill in the
		# distances for the newly created first centroid
		cent_dists = np.zeros(shape=(len(points), self.num_clusters), dtype=float)
		for i in range(len(points)):
			cent_dists[i, 0] = self._calc_euclidean_distance(points[i], centroids[0])
		# Select the rest of the centroids
		for j in range(1, self.num_clusters):
			# Pick new centroid
			new_cent = np.random.choice(range(len(points)), p = cent_dists[:, j-1] / np.sum(cent_dists[:, j-1]))
			centroids = np.vstack([centroids, points[new_cent]])
			# Compute distances from each point to the new centroid
			for i in range(len(points)):
				cent_dists[i, j] = self._calc_euclidean_distance(points[i], centroids[j])
		return centroids

	def _do_pc(self, points, centroids):
		"""
		Private helper method for cluster(). Do partition clustering to form the specified number of clusters.

		Parameters
		----------
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).
		centroids : np.array of Centroids
			Centroids initialized using kmeans++ initialization.

		Returns
		-------
		c : np.array of Clusters
			Clusters yielded from partition clustering of points.
		"""
		# Set counter for number of iterations
		iter_num = 0
		# Set boolean for whether a convergence has been reached
		converged = False
		# Set boolean for whether there's an empty cluster
		empty_cluster = False
		# Initialize empty clusters
		c = np.array([Cluster() for i in range(len(centroids))])
		# Do kmeans clustering
		while iter_num < self.max_iterations and not converged:
			# Update iteration number
			iter_num += 1
			# Calculate distances between each point and each centroid
			# Rows (i) are points and columns (j) are centroids
			cent_dists = np.zeros(shape=(len(points), len(centroids)), dtype=float)
			for j in range(len(centroids)):
				for i in range(len(points)):
					cent_dists[i, j] = self._calc_euclidean_distance(points[i], centroids[j])
			# Update clusters
			for i in range(len(points)):
				idx = np.argmin(cent_dists[i, :])
				c[idx].members = np.append(c[idx].members, i)
			# Add points to any empty clusters
			# NEED TO FILL THIS IN
			# Update centroids
			old_centroids = centroids
			centroids = np.zeros(shape=(len(c), len(points[0])), dtype=float)
			for j in range(len(c)):
				new_cent = np.sum([points[p] for p in c[j].members], axis=0) / len(c[j].members)
				centroids[j, :] = new_cent
			if np.array_equal(old_centroids, centroids):
				converged = True
		return c

	def cluster(self, points):
		"""
		Clusters a set of points using partition clustering.

		Parameters
		----------
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		c : np.array of Clusters
			Clusters yielded from partition clustering of points.
		"""
		# Initialize centroids
		centroids = self._init_centroids(points)
		# Do partition clustering
		c = self._do_pc(points, centroids)
		# Save clusters as a class attribute
		self.clusters = c
		# Also return clusters
		return c

### EVALUATING CLUSTERS ###

def get_cluster_assignments(c):
	"""
	Gets the cluster assignments for points that have already been clustered.

	Parameters
	----------
	c : np.array of Clusters
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

ligs = load_ligands("ligand_information.csv")
lig_subset = ligs[0:50]
lig_coords = np.array([l.bit_string for l in lig_subset])
pc = PartitionClustering(5, 100)
clusters = pc.cluster(lig_coords)
#print([cl.members for cl in clusters])
cl_nums = get_cluster_assignments(clusters)
#print([cl_nums])