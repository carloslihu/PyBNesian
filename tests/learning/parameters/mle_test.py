import numpy as np
import pybnesian as pbn
import pytest
from helpers.data import DATA_SIZE, generate_normal_data

df = generate_normal_data(DATA_SIZE)


def numpy_fit_mle_lg(data, variable, evidence):
    if isinstance(variable, str):
        node_data = data.loc[:, [variable] + evidence].dropna()
        variable_data = node_data.loc[:, variable]
        evidence_data = node_data.loc[:, evidence]
    else:
        node_data = data.iloc[:, [variable] + evidence].dropna()
        variable_data = node_data.iloc[:, 0]
        evidence_data = node_data.iloc[:, 1:]

    N = variable_data.shape[0]
    d = evidence_data.shape[1]
    linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
    (beta, res, _, _) = np.linalg.lstsq(
        linregress_data, variable_data.to_numpy(), rcond=None
    )
    var = res / (N - d - 1)

    return beta, var


def test_mle_create():
    with pytest.raises(ValueError) as ex:
        pbn.MLE(pbn.CKDEType())
    assert "MLE not available" in str(ex.value)

    mle = pbn.MLE(pbn.LinearGaussianCPDType())
    assert mle is not None


def test_mle_lg():
    mle = pbn.MLE(pbn.LinearGaussianCPDType())

    p = mle.estimate(df, "A", [])
    np_beta, np_var = numpy_fit_mle_lg(df, "A", [])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)

    p = mle.estimate(df, "B", ["A"])
    np_beta, np_var = numpy_fit_mle_lg(df, "B", ["A"])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)

    p = mle.estimate(df, "C", ["A", "B"])
    np_beta, np_var = numpy_fit_mle_lg(df, "C", ["A", "B"])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)

    p = mle.estimate(df, "D", ["A", "B", "C"])
    np_beta, np_var = numpy_fit_mle_lg(df, "D", ["A", "B", "C"])
    assert np.all(np.isclose(p.beta, np_beta))
    assert np.isclose(p.variance, np_var)
