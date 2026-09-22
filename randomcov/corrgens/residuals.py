from randomcov.corrutil.isvalidcorr import is_valid_corr
from randomcov.covutil.nearestposdef import nearest_positive_def
from randomcov.corrgens.wishart import wishart_corr
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def residuals_corr(n, noise=0.1, rng=None):
    # Use correlation between out of sample model errors
    # 1. Use wishart(m) to general latent_corr for m = int(math.sqrt(n+10))
    # 2. Generate X using latent_corr
    # 3. Generate random coefficients (a true linear model y = a0*X0 + a1*X1 etc
    #    plus observation noise -- without it the fit is exact and the
    #    "residuals" are floating-point rounding error
    # 4. Fit n different regression models but each time, remove half the sample randomly
    # 5. Make predictions out of sample (generate more true X)
    # 6. Compute the correlation between the model prediction errors

    # Step 1: Use wishart(m) to generate latent_corr for m = int(math.sqrt(n + 10))
    rng = np.random.default_rng(rng)
    m = int(math.sqrt(n + 10))
    latent_corr = wishart_corr(m, rng=rng)

    # Step 2: Generate X using latent_corr
    mean_vector = np.zeros(m)
    N = 1000  # Sample size
    X = rng.multivariate_normal(mean=mean_vector, cov=latent_corr, size=N)

    # Step 3: Generate random coefficients (a true linear model y = a0*X0 + a1*X1 + ...)
    coefficients = rng.standard_normal(m)
    signal = X @ coefficients
    # Observation noise, as a fraction of the signal's own scale. Each model
    # fits a different half of the sample, so the models differ and their
    # out-of-sample errors are genuine estimation error. The test targets stay
    # noiseless: shared test noise would swamp everything (mean |rho| -> 0.99).
    #
    # Note the ensemble is intrinsically low rank, and no choice of noise
    # changes that. Every error vector is y_test - X_test @ beta_hat_i, whose
    # only varying part lies in the column span of X_test, so all n of them sit
    # in an (m+1)-dimensional space with m = int(sqrt(n+10)). At n=30 that is
    # rank 6, and the matrix needs the positive-semidefinite floor. Audits that
    # invert it are therefore reading that floor, much as they do for `walk`.
    y = signal + noise * signal.std() * rng.standard_normal(N)

    # Step 5: Make predictions out of sample (generate more true X)
    X_test = rng.multivariate_normal(mean=mean_vector, cov=latent_corr, size=N)
    y_test = X_test @ coefficients  # Compute true y for test data

    residuals = []

    # Step 4: Fit n different regression models, each time removing half the sample randomly
    for i in range(n):
        # Remove half the sample randomly
        X_train, X_removed, y_train, y_removed = train_test_split(
            X, y, test_size=0.5, random_state=i)

        # Fit a regression model on the remaining data
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = model.predict(X_test)

        # Compute prediction errors
        errors = y_test - y_pred
        residuals.append(errors)

    # Step 6: Compute the correlation between the model prediction errors
    residuals_df = pd.DataFrame(residuals).T  # Each column represents a model's errors
    corr_matrix = residuals_df.corr()

    # Validate the correlation matrix
    if not is_valid_corr(corr_matrix.values):
        # Adjust to nearest positive definite matrix
        adjusted_corr = nearest_positive_def(corr_matrix.values)
        # Re-validate
        if is_valid_corr(adjusted_corr):
            corr_matrix = pd.DataFrame(adjusted_corr, index=corr_matrix.index, columns=corr_matrix.columns)
        else:
            raise ValueError("Adjusted correlation matrix is still invalid.")

    return corr_matrix.values


if __name__=='__main__':
    print(residuals_corr(n=50))