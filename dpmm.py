import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logsumexp, gammaln
from collections import Counter
from tqdm import tqdm


class DirichletProcessMixtureModel:
    """
    Dirichlet Process Mixture Model (DPMM) with Gaussian components using collapsed Gibbs sampling for inference. 
    Normal-Inverse-Wishart priors for the Gaussian components.
    """
    def __init__(self, alpha=1.0, mu_0=None, kappa_0=1.0, nu_0=None, lambda_0=None):
        self.alpha = alpha
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.nu_0 = nu_0
        self.lambda_0 = lambda_0
        
        self.cluster_assignments = None
        self.n_clusters = 0
        self.cluster_params = {}
        self.cluster_counts = Counter()
        self.data = None
        self.d = None  # dimensionality
        self.n = None  # number of data points
        
    def _initialize_hyperparameters(self, X):
        """
        Initialize hyperparameters.
        """
        self.d = X.shape[1]
        self.n = X.shape[0]
        
        if self.mu_0 is None:
            self.mu_0 = np.zeros(self.d)
        
        if self.nu_0 is None:
            self.nu_0 = self.d + 2
            
        if self.lambda_0 is None:
            self.lambda_0 = np.eye(self.d)
    
    def _log_predictive_likelihood(self, x, cluster_id):
        """
        Compute the log predictive likelihood of a data point under the cluster_id-th cluster.
        
        This uses the posterior predictive distribution, which is a multivariate t-distribution
        for the Normal-Inverse-Wishart prior.
        """
        if cluster_id not in self.cluster_params:
            # New cluster - use prior
            mu_n = self.mu_0
            kappa_n = self.kappa_0
            nu_n = self.nu_0
            lambda_n = self.lambda_0
        else:
            # Old cluster - use posterior
            mu_n, kappa_n, nu_n, lambda_n = self.cluster_params[cluster_id]
        
        mu = mu_n
        sigma = lambda_n * (kappa_n + 1) / (kappa_n * (nu_n - self.d + 1))
        df = nu_n - self.d + 1
        
        return stats.multivariate_t.logpdf(x, loc=mu, shape=sigma, df=df)
    
    def _sample_cluster_assignment(self, i):
        """
        Sample a cluster assignment for data point at i-th index using collapsed Gibbs sampling.
        """
        x = self.data[i]
        current_cluster = self.cluster_assignments[i]
        
        if current_cluster is not None:
            self.cluster_counts[current_cluster] -= 1
            if self.cluster_counts[current_cluster] == 0:
                del self.cluster_counts[current_cluster]
                del self.cluster_params[current_cluster]
        
        log_probs = []
        cluster_ids = list(self.cluster_counts.keys())
        
        for cluster_id in cluster_ids:
            # CRP prior probability
            log_prior = np.log(self.cluster_counts[cluster_id])
            # likelihood
            log_likelihood = self._log_predictive_likelihood(x, cluster_id)
            log_probs.append(log_prior + log_likelihood)
        
        log_prior_new = np.log(self.alpha)
        log_likelihood_new = self._log_predictive_likelihood(x, None)  # New cluster
        log_probs.append(log_prior_new + log_likelihood_new)
        
        log_probs = np.array(log_probs)
        log_probs -= logsumexp(log_probs)
        probs = np.exp(log_probs)
        
        if len(cluster_ids) == 0:
            new_cluster = 0  # First cluster
        else:
            choice = np.random.choice(len(probs), p=probs)
            if choice == len(cluster_ids):  # New cluster
                new_cluster = max(cluster_ids) + 1 if cluster_ids else 0
            else:
                new_cluster = cluster_ids[choice]
        
        self.cluster_assignments[i] = new_cluster
        self.cluster_counts[new_cluster] += 1
        
        self._update_cluster_params(new_cluster)
        
        return new_cluster
    
    def _update_cluster_params(self, cluster_id):
        """
        Update the posterior parameters for a cluster.
        """
        cluster_points = self.data[self.cluster_assignments == cluster_id]
        n_points = len(cluster_points)
        
        if n_points == 0:
            return
        
        x_bar = np.mean(cluster_points, axis=0)
        S = np.zeros((self.d, self.d))
        if n_points > 1:
            centered = cluster_points - x_bar
            S = centered.T @ centered
        
        # Update posterior parameters
        kappa_n = self.kappa_0 + n_points
        nu_n = self.nu_0 + n_points
        mu_n = (self.kappa_0 * self.mu_0 + n_points * x_bar) / kappa_n
        
        # Update lambda (scale matrix)
        mu_diff = x_bar - self.mu_0
        lambda_n = self.lambda_0 + S + (self.kappa_0 * n_points / kappa_n) * np.outer(mu_diff, mu_diff)
        
        self.cluster_params[cluster_id] = (mu_n, kappa_n, nu_n, lambda_n)
    
    def fit(self, X, n_iter=1000, burn_in=500, thin=1, verbose=True):
        """
        Fit the Dirichlet Process Mixture Model to the data (X) using Gibbs sampling.
        """
        self.data = np.asarray(X)
        self._initialize_hyperparameters(X)
        
        self.n = X.shape[0]
        self.cluster_assignments = np.zeros(self.n, dtype=int)
        
        self.cluster_counts = Counter({0: self.n})
        self._update_cluster_params(0)
        
        self.samples = []
        
        # Gibbs sampling
        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc="DPMM Gibbs Sampling")
            
        for iter_idx in iterator:
            for i in range(self.n):
                self._sample_cluster_assignment(i)
            
            if iter_idx >= burn_in and (iter_idx - burn_in) % thin == 0:
                self.samples.append(self.cluster_assignments.copy())
                
            if verbose and iter_idx % 100 == 0:
                n_clusters = len(self.cluster_counts)
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({"n_clusters": n_clusters})
        
        self.cluster_assignments = self.samples[-1]
        self.n_clusters = len(set(self.cluster_assignments))
        
        return self
    
    def predict(self, X_new):
        """
        Predict cluster labels for new data points in X_new.
        """
        X_new = np.asarray(X_new)
        n_new = X_new.shape[0]
        predictions = np.zeros(n_new, dtype=int)
        
        for i in range(n_new):
            x = X_new[i]
            
            log_probs = []
            cluster_ids = list(self.cluster_counts.keys())
            
            for cluster_id in cluster_ids:
                # CRP prior probability
                log_prior = np.log(self.cluster_counts[cluster_id])
                # Likelihood
                log_likelihood = self._log_predictive_likelihood(x, cluster_id)
                log_probs.append(log_prior + log_likelihood)
            
            log_prior_new = np.log(self.alpha)
            log_likelihood_new = self._log_predictive_likelihood(x, None)
            log_probs.append(log_prior_new + log_likelihood_new)
            
            log_probs = np.array(log_probs)
            log_probs -= logsumexp(log_probs)
            
            # Choose the most likely cluster
            max_idx = np.argmax(log_probs)
            if max_idx == len(cluster_ids):  # New cluster
                predictions[i] = -1  # Indicate a new cluster
            else:
                predictions[i] = cluster_ids[max_idx]
        
        return predictions
    
    def posterior_predictive(self, X_new, n_samples=100):
        """
        Compute the posterior predictive distribution for new data points.
        """
        X_new = np.asarray(X_new)
        n_new = X_new.shape[0]
        
        posterior_samples = self.samples[-n_samples:]
        
        predictions = np.zeros((n_new, len(posterior_samples)), dtype=int)
        
        for s_idx, sample in enumerate(posterior_samples):
            orig_assignments = self.cluster_assignments.copy()
            self.cluster_assignments = sample
            
            self.cluster_counts = Counter(sample)
            for cluster_id in set(sample):
                self._update_cluster_params(cluster_id)
            
            for i in range(n_new):
                x = X_new[i]
                
                log_probs = []
                cluster_ids = list(self.cluster_counts.keys())
                
                for cluster_id in cluster_ids:
                    # CRP prior probability
                    log_prior = np.log(self.cluster_counts[cluster_id])
                    # Likelihood
                    log_likelihood = self._log_predictive_likelihood(x, cluster_id)
                    log_probs.append(log_prior + log_likelihood)
                
                log_prior_new = np.log(self.alpha)
                log_likelihood_new = self._log_predictive_likelihood(x, None)
                log_probs.append(log_prior_new + log_likelihood_new)
                
                log_probs = np.array(log_probs)
                log_probs -= logsumexp(log_probs)
                probs = np.exp(log_probs)
                
                if len(cluster_ids) == 0:
                    predictions[i, s_idx] = -1  # New cluster
                else:
                    choice = np.random.choice(len(probs), p=probs)
                    if choice == len(cluster_ids):  # New cluster
                        predictions[i, s_idx] = -1
                    else:
                        predictions[i, s_idx] = cluster_ids[choice]
            
            self.cluster_assignments = orig_assignments
            
        return predictions
    
    def plot_clusters(self, X=None, dims=[0, 1], figsize=(10, 8), alpha=0.7):
        """
        Plot the clusters discovered by the model.
        """
        if X is None:
            X = self.data
            assignments = self.cluster_assignments
        else:
            X = np.asarray(X)
            assignments = self.predict(X)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        unique_clusters = np.unique(assignments)
        for cluster in unique_clusters:
            if cluster == -1:  # New cluster prediction
                continue
            mask = assignments == cluster
            ax.scatter(X[mask, dims[0]], X[mask, dims[1]], alpha=alpha, label=f'Cluster {cluster}')
        
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title('DPMM Clustering Results')
        ax.legend()
        
        return fig
    
    def plot_posterior_trace(self, figsize=(10, 6)):
        """
        Plot the trace of the number of clusters over MCMC iterations.
        """
        n_clusters = [len(set(sample)) for sample in self.samples]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(n_clusters)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Posterior Trace of Number of Clusters')
        
        return fig
    
    def plot_cluster_sizes(self, figsize=(10, 6)):
        """
        Plot the distribution of cluster sizes.
        """
        cluster_sizes = list(self.cluster_counts.values())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(range(len(cluster_sizes)), sorted(cluster_sizes, reverse=True))
        ax.set_xlabel('Cluster Rank')
        ax.set_ylabel('Cluster Size')
        ax.set_title('Distribution of Cluster Sizes')
        
        return fig
    
    def plot_pairwise_features(self, X=None, max_dims=5, figsize=(12, 10)):
        """
        Plot pairwise feature relationships colored by cluster assignment.
        """
        if X is None:
            X = self.data
            assignments = self.cluster_assignments
        else:
            X = np.asarray(X)
            assignments = self.predict(X)
        
        d = X.shape[1]
        # d = min(X.shape[1], max_dims)
        X_plot = X[:, :d]
        
        unique_clusters = np.unique(assignments)
        n_clusters = len(unique_clusters)
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
        
        fig, axes = plt.subplots(d, d, figsize=figsize)
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        
        cluster_to_color = {cluster: i for i, cluster in enumerate(unique_clusters)}
        
        for i in range(d):
            for j in range(d):
                ax = axes[i, j]
                
                ax.tick_params(axis='both', which='both', labelsize=8)
                
                if i == j:  
                    for cluster in unique_clusters:
                        mask = assignments == cluster
                        ax.hist(X_plot[mask, i], bins=15, alpha=0.5, 
                                color=colors[cluster_to_color[cluster]], 
                                density=True, label=f'Cluster {cluster}')
                    
                    if j == 0:
                        ax.set_ylabel(f'Feature {i}', fontsize=10)
                    else:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
                    
                    if i == d-1:
                        ax.set_xlabel(f'Feature {j}', fontsize=10)
                    else:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])
                        
                else: 
                    for cluster in unique_clusters:
                        mask = assignments == cluster
                        ax.scatter(X_plot[mask, j], X_plot[mask, i], 
                                  c=[colors[cluster_to_color[cluster]]], 
                                  alpha=0.6, s=20, label=f'Cluster {cluster}')
                    
                    if j == 0:
                        ax.set_ylabel(f'Feature {i}', fontsize=10)
                    else:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])
                    
                    if i == d-1:
                        ax.set_xlabel(f'Feature {j}', fontsize=10)
                    else:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])
        
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), 
                  fontsize=10, frameon=True)
        
        fig.suptitle('Pairwise Feature Relationships by Cluster', y=1.02, fontsize=14)
        plt.tight_layout()
        
        return fig



def generate_mixture_data(n_samples=500, n_components=3, random_state=None):
    """
    Makes example data to test DirichletProcessMixtureModel.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    weights = np.random.dirichlet(np.ones(n_components))
    
    means = np.random.randn(n_components, 2) * 5
    covs = np.array([np.eye(2) * (0.5 + np.random.rand()) for _ in range(n_components)])
    
    X = np.zeros((n_samples, 2))
    t = np.zeros(n_samples, dtype=int)
    
    component_assignments = np.random.choice(n_components, size=n_samples, p=weights)
    for i in range(n_samples):
        comp = component_assignments[i]
        X[i] = np.random.multivariate_normal(means[comp], covs[comp])
        t[i] = comp
    
    return X, t


def compute_cluster_purity(t, y):
    """
    Returns the purity score (0-1) of clustering labels.
    """
    contingency = np.zeros((len(set(t)), len(set(y))))
    
    for i in range(len(t)):
        contingency[t[i], y[i]] += 1
    
    return np.sum(np.max(contingency, axis=0)) / len(t)
