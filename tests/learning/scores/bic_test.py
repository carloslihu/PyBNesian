import numpy as np
import pybnesian as pbn
from scipy.stats import norm

from data import generate_normal_data

SIZE = 10000

df = generate_normal_data(SIZE)


def numpy_local_score(data, variable, evidence):
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

    means = beta[0] + np.sum(beta[1:] * evidence_data, axis=1)
    loglik = norm.logpdf(variable_data, means, np.sqrt(var))

    return loglik.sum() - np.log(N) * 0.5 * (d + 2)


def test_bic_local_score():
    gbn = pbn.GaussianNetwork(
        ["A", "B", "C", "D"],
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
    )

    bic = pbn.BIC(df)

    assert np.isclose(bic.local_score(gbn, "A", []), numpy_local_score(df, "A", []))
    assert np.isclose(
        bic.local_score(gbn, "B", ["A"]), numpy_local_score(df, "B", ["A"])
    )
    assert np.isclose(
        bic.local_score(gbn, "C", ["A", "B"]), numpy_local_score(df, "C", ["A", "B"])
    )
    assert np.isclose(
        bic.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(df, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        bic.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(df, "D", ["B", "C", "A"]),
    )

    assert bic.local_score(gbn, "A") == bic.local_score(gbn, "A", gbn.parents("A"))
    assert bic.local_score(gbn, "B") == bic.local_score(gbn, "B", gbn.parents("B"))
    assert bic.local_score(gbn, "C") == bic.local_score(gbn, "C", gbn.parents("C"))
    assert bic.local_score(gbn, "D") == bic.local_score(gbn, "D", gbn.parents("D"))


def test_bic_local_score_null():
    gbn = pbn.GaussianNetwork(
        ["A", "B", "C", "D"],
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
    )

    np.random.seed(0)
    a_null = np.random.randint(0, SIZE, size=100)
    b_null = np.random.randint(0, SIZE, size=100)
    c_null = np.random.randint(0, SIZE, size=100)
    d_null = np.random.randint(0, SIZE, size=100)

    df_null = df.copy()
    df_null.loc[df_null.index[a_null], "A"] = np.nan
    df_null.loc[df_null.index[b_null], "B"] = np.nan
    df_null.loc[df_null.index[c_null], "C"] = np.nan
    df_null.loc[df_null.index[d_null], "D"] = np.nan

    bic = pbn.BIC(df_null)

    assert np.isclose(
        bic.local_score(gbn, "A", []), numpy_local_score(df_null, "A", [])
    )
    assert np.isclose(
        bic.local_score(gbn, "B", ["A"]), numpy_local_score(df_null, "B", ["A"])
    )
    assert np.isclose(
        bic.local_score(gbn, "C", ["A", "B"]),
        numpy_local_score(df_null, "C", ["A", "B"]),
    )
    assert np.isclose(
        bic.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(df_null, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        bic.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(df_null, "D", ["B", "C", "A"]),
    )

    assert bic.local_score(gbn, "A") == bic.local_score(gbn, "A", gbn.parents("A"))
    assert bic.local_score(gbn, "B") == bic.local_score(gbn, "B", gbn.parents("B"))
    assert bic.local_score(gbn, "C") == bic.local_score(gbn, "C", gbn.parents("C"))
    assert bic.local_score(gbn, "D") == bic.local_score(gbn, "D", gbn.parents("D"))


def test_bic_score():
    gbn = pbn.GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    bic = pbn.BIC(df)

    assert np.isclose(
        bic.score(gbn),
        (
            bic.local_score(gbn, "A", [])
            + bic.local_score(gbn, "B", ["A"])
            + bic.local_score(gbn, "C", ["A", "B"])
            + bic.local_score(gbn, "D", ["A", "B", "C"])
        ),
    )
