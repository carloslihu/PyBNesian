import itertools

import numpy as np
import pandas as pd
from pybnesian import KMutualInformation, LinearCorrelation, MutualInformation
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from data import generate_normal_data, generate_normal_data_independent

SIZE = 10000
SEED = 0
data = generate_normal_data(SIZE, SEED)


def test_linear_correlation():
    df = generate_normal_data(SIZE, SEED)[["A", "B"]]

    correlations = {}
    columns = df.columns.tolist()
    for col_a, col_b in itertools.combinations(columns, 2):
        correlations[col_a + "__" + col_b] = pearsonr(
            df.loc[:, col_a], df.loc[:, col_b]
        )
    result = pd.DataFrame.from_dict(correlations, orient="index")
    result.columns = ["PCC", "p-value"]

    linear_correlation_pvalue = LinearCorrelation(df).pvalue("A", "B")
    np.testing.assert_allclose(
        np.array([result.loc["A__B", "PCC"]]),
        np.array([df.corr().loc["A", "B"]]),
        rtol=1e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.array([linear_correlation_pvalue]),
        np.array([result.loc["A__B", "p-value"]]),
        rtol=1e-5,
        atol=1e-8,
    )


def test_mutual_info():
    n_neighbors = 3
    mutual_info = MutualInformation(data).mi("A", "B")
    k_mutual_info = KMutualInformation(data, k=n_neighbors).mi("A", "B")
    sklearn_mutual_info = mutual_info_regression(
        data[["A"]], data["B"], n_neighbors=n_neighbors
    )
    # print("\n", sklearn_mutual_info[0])
    # print(mutual_info)
    # print(k_mutual_info)
    # np.testing.assert_allclose(
    #     sklearn_mutual_info,
    #     np.array([k_mutual_info]),
    #     rtol=1e-5,
    #     atol=1e-8,
    # )
