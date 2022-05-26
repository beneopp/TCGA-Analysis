import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def assign_random_cluster(num_of_features):
    cluster = np.random.rand(num_of_features)
    range_of_data = scipy.stats.iqr(matrix, axis=0)
    cluster *= range_of_data
    cluster += np.quantile(matrix, 0.25, axis=0)
    return cluster

def generate_random_clusters(num_of_clusters):
    num_of_features = matrix.shape[1]
    clusters = np.zeros((num_of_clusters, num_of_features))
    for i in range(len(clusters)):
        clusters[i, :] = assign_random_cluster(num_of_features)
    return clusters


def assign_clusters(clusters):
    num_of_samples = matrix.shape[0]
    cluster_assignments = np.array([-1] * num_of_samples)
    for i, sample in enumerate(matrix):
        sample = np.reshape(sample, (1, -1))
        sample_repeats = np.repeat(sample, len(clusters), axis=0)
        distances = np.linalg.norm(sample_repeats - clusters, axis=1)
        cluster_assignments[i] = np.argmin(distances)
    return cluster_assignments


def recenter_clusters(clusters, cluster_assignments):
    new_clusters = np.zeros((len(clusters), matrix.shape[1]))
    for i, cluster in enumerate(clusters):
        assigned_vals = matrix[cluster_assignments == i]
        num_of_assigned_vals = assigned_vals.shape[0]

        if num_of_assigned_vals == 0:
            clusters[i, :] = assign_random_cluster(matrix.shape[1])
            continue

        new_cluster = np.sum(assigned_vals, axis=0) / num_of_assigned_vals
        new_clusters[i, :] = new_cluster
    return new_clusters


def first_iteration(num_of_clusters):
    first_clusters = generate_random_clusters(num_of_clusters)
    cluster_assignments = assign_clusters(first_clusters)
    new_clusters = recenter_clusters(first_clusters, cluster_assignments)
    new_cluster_assignments = assign_clusters(new_clusters)
    return cluster_assignments, new_cluster_assignments, first_clusters, new_clusters


def calculate_objective(clusters, cluster_assignments):
    objective = 0
    for i, cluster in enumerate(clusters):
        samples_in_cluster = matrix[cluster_assignments == i]
        cluster = np.reshape(cluster, (1, -1))
        difference = samples_in_cluster - cluster
        distances = np.linalg.norm(difference, axis=1)
        objective += np.sum(np.square(distances))
    return objective


def k_means(num_of_clusters):

    cluster_assignments, new_cluster_assignments, first_clusters, clusters = first_iteration(num_of_clusters)
    objective = 0
    iterations = 0
    while np.any(new_cluster_assignments != cluster_assignments):
        cluster_assignments = new_cluster_assignments
        objective = calculate_objective(clusters, cluster_assignments)
        clusters = recenter_clusters(clusters, cluster_assignments)
        new_cluster_assignments = assign_clusters(clusters)

        if iterations % 10 == 0:
            print("iterations:", iterations)

        iterations += 1

    return objective, cluster_assignments


def cluster_data(number_of_clusters):
    reorder_indexes = get_reorder_indexes(number_of_clusters)
    new_matrix = correlations[reorder_indexes]
    new_matrix = new_matrix[:, reorder_indexes]
    return new_matrix


def get_reorder_indexes(number_of_clusters):
    total_length = correlations.shape[0]
    indexes = np.zeros(total_length, int)
    start = 0
    for i in range(number_of_clusters):
        if np.count_nonzero(cluster_assignments == i) == 0:
            continue
        cluster_indexes = np.arange(total_length, dtype=int)[cluster_assignments == i]
        end = start + len(cluster_indexes)
        indexes[start:end] = cluster_indexes
        start = end
    return indexes


print("graphing initial correlations...")
correlations = np.corrcoef(matrix)


assert correlations.shape == (matrix.shape[0], matrix.shape[0])
print("plotting")
plt.imshow(correlations, cmap="plasma")
cbar = plt.colorbar()
cbar.set_label("Pearson Correlation Coefficient")
plt.title("Heatmap Before Clustering")
plt.savefig("output/heatmap_before_cluster_samples.png")
plt.clf()


print("clustering...")
num_of_clusters = 9
trials = dict()
for i in range(0, 10):
    objective, cluster_assignments = k_means(num_of_clusters)
    trials[objective] = cluster_assignments

cluster_assignments = trials[min(trials.keys())]

print("plotting data...")
sorted_correlations = cluster_data(num_of_clusters)
plt.imshow(sorted_correlations, cmap="plasma")
cbar = plt.colorbar()
cbar.set_label("Pearson Correlation Coefficient")
plt.title("Heatmap After Clustering")
plt.savefig("output/heatmap_after_cluster_samples.png")


objectives = []
for i in range(3, 50):
    print()
    print("cluster mean", i)
    print()
    objectives.append(k_means(i))


print("plotting results...")
y = objectives
x = range(3, 3 + len(objectives))
plt.plot(x, y)
plt.title("Objectives for Different K Values for Samples")
plt.ylabel("Objective Value")
plt.xlabel("K Value")
plt.savefig("k_means_samples_2.png", bbox_inches='tight')
