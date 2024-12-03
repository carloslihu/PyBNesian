import itertools

import numpy as np
import pandas as pd
from pybnesian import (
    ChiSquare,
    KMutualInformation,
    LinearCorrelation,
    MutualInformation,
    RCoT,
)
from scipy.stats import pearsonr

from data import (
    generate_discrete_data,
    generate_discrete_data_independent,
    generate_normal_data,
    generate_normal_data_independent,
)

# from sklearn.feature_selection import mutual_info_regression


SIZE = 10000
SEED = 0
data = generate_normal_data(SIZE, SEED)
independent_data = generate_normal_data_independent(SIZE, SEED)

discrete_data = generate_discrete_data(SIZE, SEED)
independent_discrete_data = generate_discrete_data_independent(SIZE, SEED)


def test_chi_square():
    """Test the chi-square independence test with discrete data"""
    chi_square = ChiSquare(discrete_data)
    independent_chi_square = ChiSquare(independent_discrete_data)

    p_value = chi_square.pvalue("A", "B")
    independent_p_value = independent_chi_square.pvalue("A", "B")

    # Check whether the p-values are below the significance level
    assert p_value < 0.05
    assert independent_p_value > 0.05


# RFE: Test true and false independence
def test_linear_correlation():
    """Test the linear correlation independence test with normal data"""
    df = data[["A", "B"]]
    independent_df = independent_data[["A", "B"]]

    # Pybnesian Linear correlation
    linear_correlation = LinearCorrelation(df)
    independent_linear_correlation = LinearCorrelation(independent_df)
    pvalue = linear_correlation.pvalue("A", "B")
    independent_pvalue = independent_linear_correlation.pvalue("A", "B")

    # scipy pearsonr correlation
    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + "__" + col_b] = pearsonr(
            df.loc[:, col_a], df.loc[:, col_b]
        )
    result = pd.DataFrame.from_dict(
        correlations, orient="index", columns=["PCC", "p-value"]
    )

    # Compare correlation values
    np.testing.assert_allclose(
        np.array([df.corr().loc["A", "B"]]),
        np.array([result.loc["A__B", "PCC"]]),
        rtol=1e-5,
        atol=1e-8,
    )
    # Compare p-values
    np.testing.assert_allclose(
        np.array([pvalue]),
        np.array([result.loc["A__B", "p-value"]]),
        rtol=1e-5,
        atol=1e-8,
    )

    # Check whether the p-values are below the significance level
    assert pvalue < 0.05
    assert independent_pvalue > 0.05


def test_mutual_info():
    """Test the mutual information independence test with normal data"""
    mutual_info = MutualInformation(data)
    independent_mutual_info = MutualInformation(independent_data)

    # Check whether the mutual information is higher when the variables are dependent
    mutual_info_value = mutual_info.mi("A", "B")
    independent_mutual_info_value = independent_mutual_info.mi("A", "B")
    assert mutual_info_value > independent_mutual_info_value

    # Check whether the p-values are below the significance level
    pvalue = mutual_info.pvalue("A", "B")
    independent_pvalue = independent_mutual_info.pvalue("A", "B")
    assert pvalue < 0.05
    assert independent_pvalue > 0.05


def test_k_mutual_info():
    """Test the k-nearest neighbors mutual information independence test with normal data"""
    n_neighbors = 3
    k_mutual_info = KMutualInformation(data, k=n_neighbors)
    independent_k_mutual_info = KMutualInformation(independent_data, k=n_neighbors)

    # Check whether the mutual information is higher when the variables are dependent
    k_mutual_info_value = k_mutual_info.mi("A", "B")
    independent_k_mutual_info_value = independent_k_mutual_info.mi("A", "B")
    assert k_mutual_info_value > independent_k_mutual_info_value

    # Check whether the p-values are below the significance level
    # NOTE: Slow execution
    pvalue = k_mutual_info.pvalue("A", "B")
    independent_pvalue = independent_k_mutual_info.pvalue("A", "B")
    assert pvalue < 0.05
    assert independent_pvalue > 0.05

    # RFE: Results vary with scikit-learn, why?

    # sklearn_k_mutual_info_value = mutual_info_regression(
    #     data[["A"]], data["B"], n_neighbors=n_neighbors
    # )[0]
    # print(k_mutual_info_value)
    # print("\n", sklearn_k_mutual_info_value)
    # np.testing.assert_allclose(
    #     sklearn_k_mutual_info_value,
    #     np.array([k_mutual_info_value]),
    #     rtol=1e-5,
    #     atol=1e-8,
    # )
    # RFE: Test alternative https://github.com/syanga/pycit


def test_rcot():
    """Test the Randomized Conditional Correlation Test (RCoT) independence test with normal data"""
    rcot = RCoT(data, random_fourier_xy=5, random_fourier_z=100)
    independent_rcot = RCoT(independent_data, random_fourier_xy=5, random_fourier_z=100)
    p_value = rcot.pvalue("A", "B")
    independent_p_value = independent_rcot.pvalue("A", "B")

    # Check whether the p-values are below the significance level
    assert p_value < 0.05
    assert independent_p_value > 0.05
