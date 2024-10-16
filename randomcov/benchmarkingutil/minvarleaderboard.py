
from sklearn.covariance import LedoitWolf, MinCovDet
import numpy as np
import warnings
from collections import defaultdict
from randomcov.randomcovariancematrix import random_covariance_matrix
from randomcov.covutil.geodesicinterpolation import geodesic_interpolation_towards_perfect
from randomcov.covutil.nearestposdef import nearest_positive_def


warnings.filterwarnings("ignore", message=".*covariance matrix associated to your dataset is not full rank.*")


def min_var_leaderboard(n:int,
                        ports: list,              # A list of functions taking cov matrix and returning a weight vector
                        n_data_samples: int,      # The number of samples from the true cov matrix each method sees
                        corr_method: str,         # The name of the method used to generate the true cov matrix
                        var_method: str,          # The name of the method used to generate the true variance vector
                        corr_kwargs: dict=None,   # Optional additional arguments to the method used to generate the true cov matrix
                        var_kwargs: dict=None,    # Optional additional arguments to the method used to generate the true var vector
                        n_outer_iter: int = 1000, # Number of times to generate true cov matrix
                        n_inner_iter: int = 5,    # Number of times to generate samples for each true cov matrix
                        update_interval: int = 10): # How often to update leaderboard
    """

         Running tally of ex-post portfolio variance

         Compare  w.T true_cov w  for different methods taking cov -> w

    """

    def simulate_data(corr_matrix: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Simulate data from a multivariate normal distribution with the given correlation matrix.
        """
        n_dim = corr_matrix.shape[0]
        mean = np.zeros(n_dim)  # Mean is zero for all dimensions
        data = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)
        return data

    def estimate_covariance(data: np.ndarray) -> dict:
        """
        Estimate correlation matrix using different methods:
        1. Standard sample covariance
        2. Ledoit-Wolf shrinkage
        3. Minimum Covariance Determinant (robust)
        """
        # Sample covariance method
        sample_cov =  np.cov(data, rowvar=False)

        towards_perfect_cov = geodesic_interpolation_towards_perfect(sample_cov,gamma=0.25)

        # Ledoit-Wolf shrinkage method
        lw = LedoitWolf()
        lw_cov = lw.fit(data).covariance_

        # Minimum Covariance Determinant (robust method)
        mcd = MinCovDet()
        mcd_cov = mcd.fit(data).covariance_

        return {
            'sample': sample_cov,
            'lw': lw_cov,
            'mcd': mcd_cov,
            'perfect':towards_perfect_cov
        }

    def compute_portfolio_variance(w: np.ndarray, true_corr_matrix: np.ndarray) -> float:
        """
        Compute the portfolio variance given the weights and the true correlation matrix.
        """
        return np.array(w).T @ true_corr_matrix @ np.array(w)

    # Dictionary to track variances for each portfolio method
    variances = defaultdict(list)

    # Run through multiple iterations
    for iteration in range(n_outer_iter):
        # Step 1: Generate true covariance matrix
        true_cov_matrix = random_covariance_matrix(n=n, corr_method=corr_method, var_method=var_method, corr_kwargs=corr_kwargs, var_kwargs=var_kwargs)

        # Perform multiple rounds of data simulation and portfolio evaluation
        for _ in range(n_inner_iter):  # Repeat multiple times per iteration
            # Step 2: Simulate data based on the true covariance matrix
            data = simulate_data(true_cov_matrix, n_data_samples)

            # Step 3: Estimate correlation matrices from the simulated data
            estimates = estimate_covariance(data)
            for shrinkage_method in list(estimates.keys()):
                cov_est = estimates[shrinkage_method]

                # Step 4: Evaluate each portfolio method
                for  portfolio_func in ports:
                    portfolio_method = portfolio_func.__name__.replace('_long_port','').replace('_port','')

                    # Step 4a: Compute portfolio weights using the method
                    w_portfolio = portfolio_func(cov=cov_est)

                    # Step 5: Compute true portfolio variance using the true covariance matrix
                    portfolio_variance = compute_portfolio_variance(w_portfolio, true_cov_matrix)

                    # Store the variance for this method
                    combined_name = f"{portfolio_method}-{shrinkage_method}"
                    variances[combined_name].append(portfolio_variance)

        # Periodically update and display the leaderboard
        if (iteration + 1) % update_interval == 0:
            update_leaderboard(variances, iteration + 1)


def update_leaderboard(variances: dict, iteration: int):
    """
    Update and print the leaderboard of portfolio methods based on their variances.

    :param variances: Dictionary of portfolio variances tracked by method name.
    :param iteration: Current iteration number (for display purposes).
    """
    print(f"\nLeaderboard after {iteration} iterations:")

    # Compute average variance for each portfolio method
    average_variances = {method: np.mean(var_list) for method, var_list in variances.items()}

    # Sort the portfolio methods by their average variance (ascending)
    sorted_methods = sorted(average_variances.items(), key=lambda x: x[1])

    # Display the sorted leaderboard
    max_method_name_length = max(len(method) for method in average_variances.keys())
    # Calculate the maximum length for formatting

    max_method_name_length = max(len(method) for method in average_variances.keys())

    # Find the lowest average variance
    lowest_avg_var = min(avg_var for method, avg_var in sorted_methods)

    # Iterate through the sorted methods and display the relative variance
    for rank, (method, avg_var) in enumerate(sorted_methods, 1):
        relative_to_lowest = avg_var / lowest_avg_var
        print(
            f"{rank:<3}. {method:<{max_method_name_length}} : Average Variance = {avg_var:>10.4f} | Relative to lowest = {relative_to_lowest:>4.3f}")

if __name__=='__main__':
    using_precise = False
    try:
        from precise.skaters.portfoliostatic.weakport import weak_long_port       # A long only portfolio
        from precise.skaters.portfoliostatic.unitport import unit_port            # The long/short min-var portfolio
        from precise.skaters.portfoliostatic.diagport import diag_long_port       # Ignore off-diagonal entries
        using_precise = True
    except ImportError:
        print('pip install precise')

    if using_precise:
        ports = [weak_long_port, unit_port, diag_long_port]
        min_var_leaderboard(n=20, ports=ports, n_data_samples=15, corr_method='lkj', var_method='lognormal',n_inner_iter=1, update_interval=10)


