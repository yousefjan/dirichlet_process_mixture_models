import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from dpmm import DirichletProcessMixtureModel, generate_mixture_data, compute_cluster_purity
np.random.seed(410)
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 50 + "\n")

print("Generating synthetic data with 3 clusters...")
X, true_labels = generate_mixture_data(n_samples=300, n_components=3, random_state=42)

plt.figure(figsize=(10, 8))
for i in range(3):
    plt.scatter(X[true_labels == i, 0], X[true_labels == i, 1], alpha=0.7, label=f'Cluster {i}')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Synthetic Data with 3 Clusters')
plt.legend()
plt.savefig('./example/synthetic_data.png')
plt.close()
print("Saved synthetic data plot to 'synthetic_data.png'\n")

print("Fitting DPMM with alpha=1.0...")
dpmm = DirichletProcessMixtureModel(alpha=1.0)
dpmm.fit(X, verbose=True)

print(f"Number of clusters discovered: {dpmm.n_clusters}")
print(f"Cluster counts: {dict(dpmm.cluster_counts)}")

purity = compute_cluster_purity(true_labels, dpmm.cluster_assignments)
ari = adjusted_rand_score(true_labels, dpmm.cluster_assignments)
print(f"Clustering purity: {purity:.4f}")
print(f"Adjusted Rand index: {ari:.4f}")

fig = dpmm.plot_clusters()
plt.savefig('./example/dpmm_clusters.png')
plt.close()

fig = dpmm.plot_posterior_trace()
plt.savefig('./example/posterior_trace.png')
plt.close()

fig = dpmm.plot_cluster_sizes()
plt.savefig('./example/cluster_sizes.png')
plt.close()

fig = dpmm.plot_pairwise_features()
plt.savefig('./example/pairwise_features.png')
plt.close()

print("\nPredicting cluster assignments for new data...")
X_new, true_labels_new = generate_mixture_data(n_samples=50, n_components=3, random_state=43)
pred_labels = dpmm.predict(X_new)

plt.figure(figsize=(10, 8))

for cluster in np.unique(dpmm.cluster_assignments):
    mask = dpmm.cluster_assignments == cluster
    plt.scatter(X[mask, 0], X[mask, 1], alpha=0.3, label=f'Cluster {cluster} (training)')

for cluster in np.unique(pred_labels):
    if cluster == -1:  # New cluster
        plt.scatter(X_new[pred_labels == cluster, 0], X_new[pred_labels == cluster, 1], 
                    marker='x', s=100, c='k', label='New cluster')
    else:
        mask = pred_labels == cluster
        plt.scatter(X_new[mask, 0], X_new[mask, 1], marker='*', s=100, 
                    label=f'Cluster {cluster} (predicted)')

plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.title('Predicted Cluster Assignments for New Data')
plt.legend()
plt.savefig('./example/predictions.png')
plt.close()

print("=" * 50 + "\n")
