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

def calc_jaccard_distance(p1, p2):
	"""
	Calculates jaccard distance between 2 points.

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
	# If the points are the same, their distance is 0
	if np.array_equal(p1, p2):
		return 0
	i = (p1 & p2).sum()
	# If their intersection is 0, their distance is 1
	if i == 0:
		return 1
	# Formula for jaccard distance is 1-(intersection/union)
	else:
		u = (p1 | p2).sum()
		d = 1 - (i / u)
		return d

def calc_euclidean_distance(p1, p2):
	"""
	Calculates the euclidean distance between two points.

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

class HierarchicalClustering():
	"""
	Implements hierarchical clustering for an array of coordinates.

	Attributes
	----------
	num_clusters : int
		Number of clusters.
	dist_fun : function
		Function used to calculate distance.
	clusters : np.array of Clusters
		Clusters generated from doing hierarchical clustering.

	Parameters
	----------
	num_clusters : int
		Number of clusters. Saved as a class attribute.
	dist_fun : function
		Function used to calculate distance. Optional parameter. Default is jaccard distance. Saved as a class attribute.
	"""

	def __init__(self, num_clusters, dist_fun=calc_jaccard_distance):
		"""
		Initializes HierarchicalClustering object, and creates clusters.
		"""
		self.num_clusters = num_clusters
		self.dist_fun = dist_fun
		self.clusters = np.array([])

	def _init_clusters(self, points):
		"""
		Helper method for cluster() that initializes clusters.

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

	def _init_distance_matrix(self, points):
		"""
		Helper method for cluster() that initializes pairwise distance matrix for points. Only values for the upper righthand triangle are used for clustering.

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
				dm[i, j] = self.dist_fun(p1, p2)
		return dm

	def _do_hc(self, c, dm):
		"""
		Helper method for cluster() that does hierarchical clustering to form the specified number of clusters.

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

class PartitionClustering():
	"""
	Implements partition clustering for an array of coordinates.

	Attributes
	----------
	num_clusters : int
		Number of clusters.
	max_iterations : int
		Maximum number of iterations.
	dist_fun : function
		Function used to calculate distance.
	clusters : np.array of Clusters
		Clusters generated from doing hierarchical clustering.

	Parameters
	----------
	num_clusters : int
		Number of clusters. Saved as a class attribute.
	max_iterations : int
		Maximum number of iterations. Optional parameter. Default is 100. Saved as a class attribute.
	dist_fun : function
		Function used to calculate distance. Optional parameter. Default is euclidean distance. Saved as a class attribute.
	"""

	def __init__(self, num_clusters, max_iterations=100, dist_fun=calc_euclidean_distance):
		"""
		Initializes PartitionClustering object, and creates clusters.
		"""
		self.num_clusters = num_clusters
		self.max_iterations = max_iterations
		self.dist_fun = dist_fun
		self.clusters = np.array([])

	def _init_centroids(self, points):
		"""
		Initializes centroids using kmeans++ initialization.

		Parameters
		----------
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).

		Returns
		-------
		centroids : np.array of floats
			Centroids yielded from kmeans++ initialization.
		"""
		# Randomly select first centroid from points
		centroids = points[np.random.randint(len(points))]
		# Make an array representing the distance from each point to each centroid
		# where the rows are points and the columns are centroids. Fill in the
		# distances for the newly created first centroid (distances are normalized
		# by the total distances for each centroid)
		cent_dists = np.full(shape=(len(points), self.num_clusters), fill_value=np.inf, dtype=float)
		for i in range(len(points)):
			cent_dists[i, 0] = self.dist_fun(points[i], centroids[0])
		# Select the rest of the centroids
		for j in range(1, self.num_clusters):
			# For each point, find distance to nearest centroid
			near_dists = np.amin(cent_dists, axis=1)
			# Pick new centroid
			new_cent = np.random.choice(range(len(points)), p = near_dists / np.sum(near_dists))
			centroids = np.vstack([centroids, points[new_cent]])
			# Compute distances from each point to the new centroid
			for i in range(len(points)):
				cent_dists[i, j] = self.dist_fun(points[i], centroids[j])
		return centroids

	def _do_pc(self, points, centroids):
		"""
		Helper method for cluster() that does partition clustering to form the specified number of clusters.

		Parameters
		----------
		points : np.array of ints
			Points to be clustered (each row is a point and each column is a dimension/feature).
		centroids : np.array of floats
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
		# Stopping conditions are either the max number of iterations is reached
		# or the centroids stop changing
		# Reference for stopping conditions: https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
		while iter_num < self.max_iterations and not converged:
			# Update iteration number
			iter_num += 1
			# Calculate distances between each point and each centroid
			# Rows (i) are points and columns (j) are centroids
			cent_dists = np.zeros(shape=(len(points), len(centroids)), dtype=float)
			for j in range(len(centroids)):
				for i in range(len(points)):
					cent_dists[i, j] = self.dist_fun(points[i], centroids[j])
			# Update clusters
			c = np.array([Cluster() for i in range(len(centroids))])
			for i in range(len(points)):
				idx = np.argmin(cent_dists[i, :])
				c[idx].members = np.append(c[idx].members, i)
			# Add points to any empty clusters
			# Go through each cluster and if it's empty
			for j in range(len(c)):
				if len(c[j].members) == 0:
					# Then go through the other clusters and pull a point from one of them
					# if they have more than one member
					for k in range(len(c)):
						if len(c[k].members) > 1:
							c[j].members = np.append(c[j].members, c[k].members[0])
							c[k].members = np.delete(c[k].members, 0)
							break
			# Update centroids
			old_centroids = centroids
			centroids = np.zeros(shape=(len(c), len(points[0])), dtype=float)
			for j in range(len(c)):
				new_cent = np.sum([points[p] for p in c[j].members], axis=0) / len(c[j].members)
				centroids[j, :] = new_cent
			# Check for convergence
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

def compute_silhouette_score(c, points, dist_fun=calc_jaccard_distance):
	"""
	Computes the silhouette score for a clustering. This is a quality metric.

	Parameters
	----------
	c : np.array of Clusters
		Clusters to be evaluated.
	points : np.array of ints
		Coordinates of points that were clustered.
	dist_fun : function
		Function used to calculate distance. Optional parameter. Default is jaccard distance.

	Returns
	-------
	clustering_score : float
		Overall silhouette score for clustering.
	scores : dict
		Silhouette scores for each individual point, where keys are point ids and values are scores.
	"""
	# Create distance matrix up front to speed up calculations later
	point_dists = np.zeros(shape=(len(points), len(points)), dtype=float)
	for i in range(len(points)):
		for j in range(len(points)):
			point_dists[i, j] = dist_fun(points[i], points[j])
	# Make dictionary of silhouette scores for each point
	scores = {}
	# Go through each cluster
	for cl in c:
		# If there is only one member, the score is 0
		if len(cl.members) == 1:
			scores[cl.members[0]] = 0
			continue
		# Otherwise
		an = len(cl.members)
		# Go through each member (point)
		for m in cl.members:
			# Calculate a
			a_sum = 0
			# Compute distance to neighboring points (n) in cluster
			for n in cl.members:
				if n != m:
					a_sum += point_dists[m, n]
			a = a_sum / (an-1)
			# Calculate b
			# Initialize b as infinity (finalized b will be min avg distance)
			b = np.inf
			# Look at neighboring clusters (nc)
			for nc in c:
				if not np.array_equal(nc, cl):
					# Get size of nc
					ncn = len(nc.members)
					# Sum distances from m (current point) to all points in nc
					nc_sum = 0
					for nc_m in nc.members:
						nc_sum += point_dists[m, nc_m]
					nc_b = nc_sum / ncn
					# Update b
					if nc_b < b:
						b = nc_b
			# Calculate s
			s = (b-a) / max(a,b)
			# Update scores dict
			scores[m] = s
	# Compute mean score
	clustering_score = sum(scores.values()) / len(scores)
	# Return score (for clustering) and scores dict (for all points)
	return clustering_score, scores

def compute_jaccard_index(c1, c2):
	"""
	Computes the jaccard index between two clusterings. This is a metric for comparing clustering similarity.

	Parameters
	----------
	c1 : np.array of Clusters
		First clustering.
	c2 : np.array of Clusters
		Second clustering.

	Returns
	-------
	ji : float
		Jaccard index for the two clusterings.
	"""
	# Get cluster assignments for both clusters
	c1a = get_cluster_assignments(c1)
	c2a = get_cluster_assignments(c2)
	# Initialize outcome values to 0
	f11 = 0
	f10 = 0
	f01 = 0
	# Go through all pairs of points
	for p1 in c1a.keys():
		for p2 in c1a.keys():
			# Check if same clustering in c1
			if c1a[p1] == c1a[p2]:
				# Check if same clustering in c2
				if c2a[p1] == c2a[p2]:
					f11 += 1 # same in both
				else:
					f01 += 1 # same in 1, not in 2
			elif c2a[p1] == c2a[p2]:
				f10 += 1 # same in 2, not in 1
	# Calculate jaccard index
	ji = f11 / (f01 + f10 + f11)
	return ji

def update_member_ids(ids, clusters):
	"""
	For cluster members, replace the point number with the id number. Also sort cluster members (purely for aesthetics).

	Parameters
	----------
	ids : np.array of ints
		Id numbers for each point.
	clusters : np.array of Clusters
		Clusters yielded from hierarchical or partition clustering.

	Returns
	-------
	c : np.array of Clusters
		Clusters with the members containing the original id numbers in sorted order.
	"""
	# Make dict mapping point index number to original id number
	number_map = {pi : og for pi, og in enumerate(ids)}
	# Replace point numbers with original id numbers and sort them
	c = np.array([Cluster() for j in range(len(clusters))])
	for j in range(len(clusters)):
		for p in clusters[j].members:
			c[j].members = np.append(c[j].members, number_map[p])
		c[j].members = np.sort(c[j].members)
	return c