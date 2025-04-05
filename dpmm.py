import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import multivariate_normal, invwishart, multivariate_t
from tqdm import tqdm
from collections import Counter

class DirichletProcessMixtureModel:
    def __init__(self, data, alpha=1.0, mu0=None, kappa0=1.0, nu0=None, Lambda0=None):
        """
        Multivariate Dirichlet process mixture model (DPMM) with Gaussian components using collapsed Gibbs sampling for inference. 
        """
        self.data = np.asarray(data)
        self.n, self.d = self.data.shape
        
        self.alpha = alpha
        
        if mu0 is None:
            self.mu0 = np.mean(self.data, axis=0)
        else:
            self.mu0 = mu0
            
        self.kappa0 = kappa0
        
        if nu0 is None:
            self.nu0 = self.d + 2
        else:
            self.nu0 = nu0
            
        if Lambda0 is None:
            self.Lambda0 = np.eye(self.d)
        else:
            self.Lambda0 = Lambda0
        
        # We start with each point in its own cluster.
        self.z = np.arange(self.n)
        # {cluster_label: (mu, Sigma)}
        self.params = {i: (self.data[i], np.eye(self.d)) for i in range(self.n)}
        
        self.cluster_history = [len(np.unique(self.z))]
        self.assignment_history = [self.z.copy()]
        
        self.n_clusters = len(np.unique(self.z))
        self.cluster_counts = Counter(self.z)
        self.cluster_assignments = {}
    
    def _posterior_niw(self, X):
        """
        Compute the posterior NIW parameters given data X assigned to a cluster.
        X: array of shape (n_k, d)
        Returns: (mu_n, kappa_n, nu_n, Lambda_n)
        """
        n_k = X.shape[0]
        if n_k == 0:
            raise ValueError("No data in cluster for NIW update.")
        
        x_bar = np.mean(X, axis=0)
        S = np.zeros((self.d, self.d))
        if n_k > 1:
            X_centered = X - x_bar
            S = X_centered.T @ X_centered
        
        kappa_n = self.kappa0 + n_k
        mu_n = (self.kappa0 * self.mu0 + n_k * x_bar) / kappa_n
        nu_n = self.nu0 + n_k
        # The updated scale matrix for the Inverse-Wishart
        diff = (x_bar - self.mu0).reshape(-1, 1)
        Lambda_n = self.Lambda0 + S + (self.kappa0 * n_k / kappa_n) * (diff @ diff.T)
        return mu_n, kappa_n, nu_n, Lambda_n
    
    def _sample_cluster_params(self, X):
        """
        Given data X for a cluster, sample new (mu, Sigma) from the NIW posterior.
        """
        mu_n, kappa_n, nu_n, Lambda_n = self._posterior_niw(X)
        # Sample covariance from Inverse-Wishart
        Sigma = invwishart.rvs(df=nu_n, scale=Lambda_n)
        # Sample mean from multivariate normal
        mu = np.random.multivariate_normal(mu_n, Sigma / kappa_n)
        return mu, Sigma
    
    def _predictive_density(self, x):
        """
        Compute the predictive density for x under a new cluster using the NIW prior.
        The predictive density is a multivariate t-distribution with:
          - location: mu0
          - shape: (kappa0+1)/(kappa0*(nu0-d+1)) * Lambda0
          - degrees of freedom: nu0-d+1
        """
        df = self.nu0 - self.d + 1
        scale = (self.kappa0 + 1) / (self.kappa0 * df) * self.Lambda0
        return multivariate_t.pdf(x, loc=self.mu0, shape=scale, df=df)
    
    def run(self, n_iter=300):
        """
        Run the Gibbs sampler for n_iter iterations.
        """
        for it in tqdm(range(n_iter), desc="DPMM Sampling"):
            for i in range(self.n):
                # Remove x_i from its cluster.
                curr_cluster = self.z[i]
                idx = np.where(self.z == curr_cluster)[0]
                idx = idx[idx != i]
                if len(idx) == 0:
                    # Remove empty cluster from parameters.
                    if curr_cluster in self.params:
                        del self.params[curr_cluster]
                self.z[i] = -1  # temporary placeholder

                # List current clusters.
                clusters = list(self.params.keys())
                probs = []

                # Compute probability for each existing cluster.
                for k in clusters:
                    # Get all data points currently assigned to cluster k.
                    X_k = self.data[self.z == k]
                    n_k = X_k.shape[0]
                    # Likelihood of x_i given cluster k parameters.
                    mu_k, Sigma_k = self.params[k]
                    likelihood = multivariate_normal.pdf(self.data[i], mean=mu_k, cov=Sigma_k)
                    probs.append(n_k * likelihood)
                
                # Probability for a new cluster (using the NIW predictive density)
                new_prob = self.alpha * self._predictive_density(self.data[i])
                probs.append(new_prob)
                
                probs = np.array(probs)
                probs /= np.sum(probs)
                # Sample a new assignment for x_i.
                choice = np.random.choice(len(probs), p=probs)
                if choice == len(clusters):
                    # Create a new cluster.
                    new_cluster_label = max(clusters) + 1 if clusters else 0
                    self.z[i] = new_cluster_label
                    # For a new cluster, use the data point alone to sample parameters.
                    self.params[new_cluster_label] = self._sample_cluster_params(np.array([self.data[i]]))
                else:
                    self.z[i] = clusters[choice]
            
            # After reassigning all points, update the parameters for each cluster.
            unique_clusters = np.unique(self.z)
            for k in unique_clusters:
                idx = np.where(self.z == k)[0]
                X_k = self.data[idx]
                self.params[k] = self._sample_cluster_params(X_k)
            
            self.cluster_history.append(len(np.unique(self.z)))
            self.assignment_history.append(self.z.copy())
        
        # Update n_clusters and cluster_counts after all iterations
        self.n_clusters = len(np.unique(self.z))
        self.cluster_counts = Counter(self.z)
        self.cluster_assignments = self.z.copy()
    
    def plot_clusters(self):
        """
        If the data is 2D, plot the clusters.
        """
        if self.d != 2:
            print("Plotting is only supported for 2D data.")
            return
        
        clusters = np.unique(self.z)
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        plt.figure(figsize=(8,6))
        for k, c in zip(clusters, colors):
            idx = np.where(self.z == k)[0]
            plt.scatter(self.data[idx, 0], self.data[idx, 1], color=c, label=f"Cluster {k}")
            # Plot an ellipse for the covariance (optional; here we mark the mean)
            mu_k, _ = self.params[k]
            plt.scatter(mu_k[0], mu_k[1], marker="x", s=200, color=c)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title("DPMM Clusters (Multivariate with NIW Prior)")
        plt.legend()
        plt.show()
    
    def plot_trace(self):
        """
        Plot the trace of the number of clusters over iterations.
        """
        plt.figure(figsize=(8,4))
        plt.plot(self.cluster_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Number of clusters")
        plt.title("Trace of Number of Clusters")
        plt.show()
    
    def plot_cluster_sizes(self):
        """
        Plot the distribution of cluster sizes.
        """
        counts = self.cluster_counts
        
        sorted_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        cluster_labels = [f"Cluster {k}" for k, _ in sorted_clusters]
        cluster_sizes = [v for _, v in sorted_clusters]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(cluster_labels)), cluster_sizes, color='skyblue')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{cluster_sizes[i]}',
                    ha='center', va='bottom')
        
        plt.xlabel('Cluster')
        plt.ylabel('Number of Data Points')
        plt.title('Distribution of Cluster Sizes')
        plt.xticks(range(len(cluster_labels)), cluster_labels, rotation=45, ha='right')
        plt.tight_layout()
        
    
    def animate_assignments(self, save_path="dpmm_niw_animation.gif", interval=300, dpi=150):
        """
        Create an animation (GIF) showing evolution of cluster assignments (only for 2D data).
        """
        if self.d != 2:
            print("Animation is only supported for 2D data.")
            return
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        title = ax.text(0.5, 1.03, "", transform=ax.transAxes, ha="center")
        scat = ax.scatter([], [], s=50)
        
        def init():
            scat.set_offsets([])
            title.set_text("")
            return scat, title
        
        def animate(i):
            assignments = self.assignment_history[i]
            clusters = np.unique(assignments)
            colors = {}
            cmap = plt.cm.tab10
            for j, k in enumerate(clusters):
                colors[k] = cmap(j % 10)
            point_colors = [colors[lab] for lab in assignments]
            offsets = self.data
            scat.set_offsets(offsets)
            scat.set_color(point_colors)
            title.set_text(f"Iteration {i}")
            return scat, title
        
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.assignment_history), interval=interval, blit=True)
        anim.save(save_path, writer="imagemagick", dpi=dpi)
        plt.close(fig)
        
    def predict(self, X_new):
        """
        Predict cluster assignments for new data points.
        """
        X_new = np.asarray(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
            
        if X_new.shape[1] != self.d:
            raise ValueError(f"New data has {X_new.shape[1]} features, but model was trained with {self.d} features.")
            
        n_new = X_new.shape[0]
        z_new = np.zeros(n_new, dtype=int)
        
        existing_clusters = list(self.params.keys())
        
        for i in range(n_new):
            probs = []
            
            for k in existing_clusters:
                mu_k, Sigma_k = self.params[k]
                # Get count of points in this cluster
                n_k = np.sum(self.z == k)
                # Compute likelihood of x_i given cluster k parameters
                likelihood = multivariate_normal.pdf(X_new[i], mean=mu_k, cov=Sigma_k)
                probs.append(n_k * likelihood)
            
            # Probability for a new cluster (using the NIW predictive density)
            new_prob = self.alpha * self._predictive_density(X_new[i])
            probs.append(new_prob)
            
            # Normalize probabilities
            probs = np.array(probs)
            probs /= np.sum(probs)
            
            # Assign to the most probable cluster
            choice = np.argmax(probs)
            if choice == len(existing_clusters):
                z_new[i] = -1
            else:
                z_new[i] = existing_clusters[choice]
                
        return z_new

def compute_cluster_purity(true_labels, cluster_assignments):
    clusters = np.unique(cluster_assignments)
    correct_assignments = 0
    
    for cluster in clusters:
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        
        cluster_true_labels = true_labels[cluster_indices]
        if len(cluster_true_labels) > 0:
            label_counts = Counter(cluster_true_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            correct_assignments += np.sum(cluster_true_labels == most_common_label)
    
    purity = correct_assignments / len(true_labels)
    
    return purity

# Example usage:
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 300
    d = 2
    centers = np.array([[0, 0], [5, 5], [-5, 5]])
    data = []
    for center in centers:
        data.append(np.random.multivariate_normal(center, np.eye(d), size=n_samples//3))
    data = np.vstack(data)
    
    model = DirichletProcessMixtureModel(data, alpha=1.0, kappa0=1.0, nu0=d+2, Lambda0=np.eye(d))
