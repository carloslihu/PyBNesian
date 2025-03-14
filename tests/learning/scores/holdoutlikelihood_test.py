import numpy as np
import pandas as pd
import pybnesian as pbn
import pytest
from helpers.data import generate_normal_data
from scipy.stats import gaussian_kde, norm

SIZE = 1000
df = generate_normal_data(SIZE)
seed = 0


def numpy_local_score(
    node_type: pbn.FactorType,
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    variable: str,
    evidence: list[str],
):

    node_data = training_data.loc[:, [variable] + evidence].dropna()
    variable_data = node_data.loc[:, variable]
    evidence_data = node_data.loc[:, evidence]
    test_node_data = test_data.loc[:, [variable] + evidence].dropna()
    test_variable_data = test_node_data.loc[:, variable]
    test_evidence_data = test_node_data.loc[:, evidence]

    loglik = 0
    if node_type == pbn.LinearGaussianCPDType():
        N = variable_data.shape[0]
        d = evidence_data.shape[1]
        linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
        (beta, res, _, _) = np.linalg.lstsq(
            linregress_data, variable_data.to_numpy(), rcond=None
        )
        var = res / (N - d - 1)

        means = beta[0] + np.sum(beta[1:] * test_evidence_data, axis=1)
        loglik = norm.logpdf(test_variable_data, means, np.sqrt(var)).sum()

    elif node_type == pbn.CKDEType():
        k_joint = gaussian_kde(
            node_data.to_numpy().T,
            bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
            * s.scotts_factor(),
        )
        if evidence:
            k_marg = gaussian_kde(evidence_data.to_numpy().T, bw_method=k_joint.factor)
            loglik = np.sum(
                k_joint.logpdf(test_node_data.to_numpy().T)
                - k_marg.logpdf(test_evidence_data.to_numpy().T)
            )
        else:
            loglik = np.sum(k_joint.logpdf(test_node_data.to_numpy().T))

    return loglik


def test_holdout_create():
    """Test HoldoutLikelihood creation with different parameters"""
    s = pbn.HoldoutLikelihood(df)
    assert s.training_data().num_rows == 0.8 * SIZE
    assert s.test_data().num_rows == 0.2 * SIZE

    s = pbn.HoldoutLikelihood(df, test_ratio=0.5)
    assert s.training_data().num_rows == 0.5 * SIZE
    assert s.test_data().num_rows == 0.5 * SIZE

    s = pbn.HoldoutLikelihood(df, test_ratio=0.2, seed=0)
    s2 = pbn.HoldoutLikelihood(df, test_ratio=0.2, seed=0)

    assert s.training_data().equals(s2.training_data())
    assert s.test_data().equals(s2.test_data())

    with pytest.raises(ValueError) as ex:
        s = pbn.HoldoutLikelihood(df, test_ratio=10, seed=0)
    assert "test_ratio must be a number" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        s = pbn.HoldoutLikelihood(df, test_ratio=0, seed=0)
    assert "test_ratio must be a number" in str(ex.value)


def test_holdout_local_score_gbn():
    gbn = pbn.GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    hl = pbn.HoldoutLikelihood(df, 0.2, seed)

    assert np.isclose(
        hl.local_score(gbn, "A", []),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "A",
            [],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "B", ["A"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "B",
            ["A"],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "C", ["A", "B"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "C",
            ["A", "B"],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "D",
            ["A", "B", "C"],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "D", ["A", "B", "C"]),
        hl.local_score(gbn, "D", ["B", "C", "A"]),
    )

    assert hl.local_score(gbn, "A") == hl.local_score(gbn, "A", gbn.parents("A"))
    assert hl.local_score(gbn, "B") == hl.local_score(gbn, "B", gbn.parents("B"))
    assert hl.local_score(gbn, "C") == hl.local_score(gbn, "C", gbn.parents("C"))
    assert hl.local_score(gbn, "D") == hl.local_score(gbn, "D", gbn.parents("D"))


def test_holdout_local_score_gbn_null():
    gbn = pbn.GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
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

    hl = pbn.HoldoutLikelihood(df_null, 0.2, seed)

    assert np.isclose(
        hl.local_score(gbn, "A", []),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "A",
            [],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "B", ["A"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "B",
            ["A"],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "C", ["A", "B"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "C",
            ["A", "B"],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "D",
            ["A", "B", "C"],
        ),
    )
    assert np.isclose(
        hl.local_score(gbn, "D", ["A", "B", "C"]),
        hl.local_score(gbn, "D", ["B", "C", "A"]),
    )

    assert hl.local_score(gbn, "A") == hl.local_score(gbn, "A", gbn.parents("A"))
    assert hl.local_score(gbn, "B") == hl.local_score(gbn, "B", gbn.parents("B"))
    assert hl.local_score(gbn, "C") == hl.local_score(gbn, "C", gbn.parents("C"))
    assert hl.local_score(gbn, "D") == hl.local_score(gbn, "D", gbn.parents("D"))


def test_holdout_local_score_spbn():
    spbn = pbn.SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
    )

    hl = pbn.HoldoutLikelihood(df, 0.2, seed)

    assert np.isclose(
        hl.local_score(spbn, "A", []),
        numpy_local_score(
            pbn.CKDEType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "A",
            [],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "B", ["A"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "B",
            ["A"],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "C", ["A", "B"]),
        numpy_local_score(
            pbn.CKDEType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "C",
            ["A", "B"],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "D",
            ["A", "B", "C"],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "D",
            ["B", "C", "A"],
        ),
    )

    assert hl.local_score(spbn, "A") == hl.local_score(spbn, "A", spbn.parents("A"))
    assert hl.local_score(spbn, "B") == hl.local_score(spbn, "B", spbn.parents("B"))
    assert hl.local_score(spbn, "C") == hl.local_score(spbn, "C", spbn.parents("C"))
    assert hl.local_score(spbn, "D") == hl.local_score(spbn, "D", spbn.parents("D"))


def test_holdout_local_score_null_spbn():
    spbn = pbn.SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
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

    hl = pbn.HoldoutLikelihood(df_null, 0.2, seed)

    assert np.isclose(
        hl.local_score(spbn, "A", []),
        numpy_local_score(
            pbn.CKDEType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "A",
            [],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "B", ["A"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "B",
            ["A"],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "C", ["A", "B"]),
        numpy_local_score(
            pbn.CKDEType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "C",
            ["A", "B"],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "D",
            ["A", "B", "C"],
        ),
    )
    assert np.isclose(
        hl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(
            pbn.LinearGaussianCPDType(),
            hl.training_data().to_pandas(),
            hl.test_data().to_pandas(),
            "D",
            ["B", "C", "A"],
        ),
    )

    assert hl.local_score(spbn, "A") == hl.local_score(spbn, "A", spbn.parents("A"))
    assert hl.local_score(spbn, "B") == hl.local_score(spbn, "B", spbn.parents("B"))
    assert hl.local_score(spbn, "C") == hl.local_score(spbn, "C", spbn.parents("C"))
    assert hl.local_score(spbn, "D") == hl.local_score(spbn, "D", spbn.parents("D"))


def test_holdout_score():
    gbn = pbn.GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    hl = pbn.HoldoutLikelihood(df, 0.2, 0)

    assert np.isclose(
        hl.score(gbn),
        (
            hl.local_score(gbn, "A", [])
            + hl.local_score(gbn, "B", ["A"])
            + hl.local_score(gbn, "C", ["A", "B"])
            + hl.local_score(gbn, "D", ["A", "B", "C"])
        ),
    )

    spbn = pbn.SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
    )

    assert np.isclose(
        hl.score(spbn),
        (
            hl.local_score(spbn, "A")
            + hl.local_score(spbn, "B")
            + hl.local_score(spbn, "C")
            + hl.local_score(spbn, "D")
        ),
    )
