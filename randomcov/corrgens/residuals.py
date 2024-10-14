from randomcov.corrgens.isvalidcorr import is_valid_corr
from randomcov.corrgens.nearestposdef import nearest_positive_def
from randomcov.corrgens.wishart import wishart_corr
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def residuals_corr(n):
    # Use correlation between out of sample model errors
    # 1. Use wishart(m) to general latent_corr for m = int(math.sqrt(n+10))
    # 2. Generate X using latent_corr
    # 3. Generate random coefficients (a true linear model y = a0*X0 + a1*X1 etc
    # 4. Fit n different regression models but each time, remove half the sample randomly
    # 5. Make predictions out of sample (generate more true X)
    # 6. Compute the correlation between the model prediction errors

    # Step 1: Use wishart(m) to generate latent_corr for m = int(math.sqrt(n + 10))
    m = int(math.sqrt(n + 10))
    latent_corr = wishart_corr(m)

    # Step 2: Generate X using latent_corr
    mean_vector = np.zeros(m)
    N = 1000  # Sample size
    X = np.random.multivariate_normal(mean=mean_vector, cov=latent_corr, size=N)

    # Step 3: Generate random coefficients (a true linear model y = a0*X0 + a1*X1 + ...)
    coefficients = np.random.randn(m)
    y = X @ coefficients  # Compute true y values

    # Step 5: Make predictions out of sample (generate more true X)
    X_test = np.random.multivariate_normal(mean=mean_vector, cov=latent_corr, size=N)
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