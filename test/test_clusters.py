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
	id2 = int(4)
	score2 = float(-2.7)
	smiles2 = "OC#N"
	on_bits2 = np.array([260, 291, 360, 674, 790, 807])
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	# Check that first ligand info is correct
	lig1 = ligs[1]
	assert lig1.id == id1
	assert lig1.score == score1
	assert lig1.smiles == smiles1
	assert np.array_equal(lig1.on_bits, on_bits1)
	# Check that second ligand info is correct
	lig2 = ligs[3]
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

# Test jaccard distance calculation
def test_jaccard_distance():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:2]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Calculate jaccard distance
	jd = algs.calc_jaccard_distance(lig_coords[0], lig_coords[1])
	# Set true value
	true_jd = float(1.0)
	# Check that they match
	assert jd == true_jd

# Test euclidean distance calculation
def test_euclidean_distance():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:2]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Calculate jaccard distance
	ed = algs.calc_euclidean_distance(lig_coords[0], lig_coords[1])
	# Set true value
	true_ed = float(2.449489742783178)
	# Check that they match
	assert ed == true_ed

# Test distance matrix for hierarchical clustering
def test_hc_distance_matrix():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:3]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Compute distance matrix
	hc = algs.HierarchicalClustering(2)
	dm = hc._init_distance_matrix(lig_coords)
	# Set true value
	true_dm = np.array([[np.inf, 1, 1], [np.inf, np.inf, 0.8], [np.inf, np.inf, np.inf]], dtype=float)
	# Check that they match
	assert np.array_equal(dm, true_dm)

# Test hierarchical clustering
def test_hierarchical():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:10]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Do clustering
	hc = algs.HierarchicalClustering(2)
	clusters = hc.cluster(lig_coords)
	# Set true values
	true_cluster1 = np.array([0, 3])
	true_cluster2 = np.array([1,2,6,5,4,9,7,8])
	# Check that they match
	assert np.array_equal(clusters[0].members, true_cluster1)
	assert np.array_equal(clusters[1].members, true_cluster2)

# Test centroid initialization (using kmeans++ initialization)
# Since the initiation is non-deterministic, I didn't test the centroids themselves,
# so I just made sure the correct number of centroids are outputted and that they
# have the correct dimensions
def test_centroid_init():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:10]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Do centroid initialization
	pc = algs.PartitionClustering(2)
	cents = pc._init_centroids(lig_coords)
	# Check that there are 2 centroids
	assert len(cents) == 2
	# Check that they have the correct dimensions
	assert len(cents[0]) == 1024
	assert len(cents[1]) == 1024

# Test partition clustering
# Since this algorithm is non-deterministic, I didn't test the clusters themselves,
# so I just made sure the correct number of clusters are outputted and none of them
# are empty
def test_partitioning():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:10]
	lig_ids = np.array([l.id for l in lig_subset])
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Do clustering
	pc = algs.PartitionClustering(5)
	clusters = pc.cluster(lig_coords)
	# Check that there are 5 clusters
	assert len(clusters) == 5
	# Check that all of the clusters are non-empty
	for c in clusters:
		assert len(c.members) > 0

# Test getting the cluster assignments from clusters
def test_cluster_assign():
	# Make clusters
	c = np.array([algs.Cluster() for j in range(2)])
	c[0].members = np.array([0,1])
	c[1].members = np.array([2])
	# Get cluster assignments
	ca = algs.get_cluster_assignments(c)
	# Set true cluster assignments
	true_ca = {0: 0, 1: 0, 2: 1}
	# Check that they match
	assert ca == true_ca

# Test computing silhouette score (for clustering quality)
def test_silhouette_score():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:10]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Do clustering
	hc = algs.HierarchicalClustering(2)
	c = hc.cluster(lig_coords)
	# Compute silhouette score
	s = algs.compute_silhouette_score(c, lig_coords)
	# Set true value
	true_s = float(0.11641284288871127)
	# Check that they match
	assert s == true_s

# Test computing jaccard index (for clustering similarity)
def test_jaccard_index():
	# Load ligands
	ligs = algs.load_ligands("test_data/test_ligands.csv")
	lig_subset = ligs[0:10]
	lig_coords = np.array([l.bit_string for l in lig_subset])
	# Do first clustering
	hc1 = algs.HierarchicalClustering(3)
	c1 = hc1.cluster(lig_coords)
	# Do second clustering
	hc2 = algs.HierarchicalClustering(5)
	c2 = hc2.cluster(lig_coords)
	# Compute jaccard index
	j = algs.compute_jaccard_index(c1, c2)
	# Set true value
	true_j = float(0.7272727272727273)
	# Check that they match
	assert j == true_j

# Test member id updating
def test_update_ids():
	# Make array of ids
	ids = np.array([0,2,4,10,666])
	# Make clusters
	c = np.array([algs.Cluster() for j in range(2)])
	c[0].members = np.array([2,4,0])
	c[1].members = np.array([1,3])
	# Update ids
	updated_c = algs.update_member_ids(ids, c)
	# Set true clusters
	true_c = np.array([algs.Cluster() for j in range(2)])
	true_c[0].members = np.array([0,4,666])
	true_c[1].members = np.array([2,10])
	# Check that they match
	assert np.array_equal(updated_c[0].members, true_c[0].members)
	assert np.array_equal(updated_c[1].members, true_c[1].members)
