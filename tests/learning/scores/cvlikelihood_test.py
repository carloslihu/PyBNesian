import numpy as np
import pandas as pd
import pybnesian as pbn
import pytest
from scipy.stats import gaussian_kde, norm
from util_test import generate_normal_data

SIZE = 1000
df = generate_normal_data(SIZE)

seed = 0


def numpy_local_score(
    node_type: pbn.FactorType, data: pd.DataFrame, variable: str, evidence: list[str]
):
    cv = pbn.CrossValidation(data, 10, seed)
    loglik = 0
    for train_df, test_df in cv:
        node_data = train_df.to_pandas().loc[:, [variable] + evidence].dropna()
        variable_data = node_data.loc[:, variable]
        evidence_data = node_data.loc[:, evidence]
        test_node_data = test_df.to_pandas().loc[:, [variable] + evidence].dropna()
        test_variable_data = test_node_data.loc[:, variable]
        test_evidence_data = test_node_data.loc[:, evidence]

        if node_type == pbn.LinearGaussianCPDType():
            N = variable_data.shape[0]
            d = evidence_data.shape[1]
            linregress_data = np.column_stack((np.ones(N), evidence_data.to_numpy()))
            (beta, res, _, _) = np.linalg.lstsq(
                linregress_data, variable_data.to_numpy(), rcond=None
            )
            var = res / (N - d - 1)

            means = beta[0] + np.sum(beta[1:] * test_evidence_data, axis=1)
            loglik += norm.logpdf(test_variable_data, means, np.sqrt(var)).sum()

        elif node_type == pbn.CKDEType():
            k_joint = gaussian_kde(
                node_data.to_numpy().T,
                bw_method=lambda s: np.power(4 / (s.d + 2), 1 / (s.d + 4))
                * s.scotts_factor(),
            )
            if evidence:
                k_marg = gaussian_kde(
                    evidence_data.to_numpy().T, bw_method=k_joint.factor
                )
                loglik += np.sum(
                    k_joint.logpdf(test_node_data.to_numpy().T)
                    - k_marg.logpdf(test_evidence_data.to_numpy().T)
                )
            else:
                loglik += np.sum(k_joint.logpdf(test_node_data.to_numpy().T))

    return loglik


def test_cvl_create():
    s = pbn.CVLikelihood(df)
    assert len(list(s.cv)) == 10
    s = pbn.CVLikelihood(df, 5)
    assert len(list(s.cv)) == 5

    s = pbn.CVLikelihood(df, 10, 0)
    assert len(list(s.cv)) == 10
    s2 = pbn.CVLikelihood(df, 10, 0)
    assert len(list(s2.cv)) == 10

    for (train_cv, test_cv), (train_cv2, test_cv2) in zip(s.cv, s2.cv):
        assert train_cv.equals(
            train_cv2
        ), "Train CV DataFrames with the same seed are not equal."
        assert test_cv.equals(
            test_cv2
        ), "Test CV DataFrames with the same seed are not equal."

    with pytest.raises(ValueError) as ex:
        s = pbn.CVLikelihood(df, SIZE + 1)
    assert "Cannot split" in str(ex.value)


def test_cvl_local_score_gbn():
    gbn = pbn.GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    cvl = pbn.CVLikelihood(df, 10, seed)

    assert np.isclose(
        cvl.local_score(gbn, "A", []),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "A", []),
    )
    assert np.isclose(
        cvl.local_score(gbn, "B", ["A"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "B", ["A"]),
    )
    assert np.isclose(
        cvl.local_score(gbn, "C", ["A", "B"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "C", ["A", "B"]),
    )
    assert np.isclose(
        cvl.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        cvl.local_score(gbn, "D", ["A", "B", "C"]),
        cvl.local_score(gbn, "D", ["B", "C", "A"]),
    )

    assert cvl.local_score(gbn, "A") == cvl.local_score(gbn, "A", gbn.parents("A"))
    assert cvl.local_score(gbn, "B") == cvl.local_score(gbn, "B", gbn.parents("B"))
    assert cvl.local_score(gbn, "C") == cvl.local_score(gbn, "C", gbn.parents("C"))
    assert cvl.local_score(gbn, "D") == cvl.local_score(gbn, "D", gbn.parents("D"))


def test_cvl_local_score_gbn_null():
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

    cvl = pbn.CVLikelihood(df_null, 10, seed)

    assert np.isclose(
        cvl.local_score(gbn, "A", []),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "A", []),
    )
    assert np.isclose(
        cvl.local_score(gbn, "B", ["A"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "B", ["A"]),
    )
    assert np.isclose(
        cvl.local_score(gbn, "C", ["A", "B"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "C", ["A", "B"]),
    )
    assert np.isclose(
        cvl.local_score(gbn, "D", ["A", "B", "C"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        cvl.local_score(gbn, "D", ["A", "B", "C"]),
        cvl.local_score(gbn, "D", ["B", "C", "A"]),
    )

    assert cvl.local_score(gbn, "A") == cvl.local_score(gbn, "A", gbn.parents("A"))
    assert cvl.local_score(gbn, "B") == cvl.local_score(gbn, "B", gbn.parents("B"))
    assert cvl.local_score(gbn, "C") == cvl.local_score(gbn, "C", gbn.parents("C"))
    assert cvl.local_score(gbn, "D") == cvl.local_score(gbn, "D", gbn.parents("D"))


def test_cvl_local_score_spbn():
    spbn = pbn.SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
    )

    cvl = pbn.CVLikelihood(df, 10, seed)

    assert np.isclose(
        cvl.local_score(spbn, "A", []), numpy_local_score(pbn.CKDEType(), df, "A", [])
    )
    assert np.isclose(
        cvl.local_score(spbn, "B", ["A"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "B", ["A"]),
    )
    assert np.isclose(
        cvl.local_score(spbn, "C", ["A", "B"]),
        numpy_local_score(pbn.CKDEType(), df, "C", ["A", "B"]),
    )
    assert np.isclose(
        cvl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        cvl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "D", ["B", "C", "A"]),
    )

    assert cvl.local_score(spbn, "A") == cvl.local_score(spbn, "A", spbn.parents("A"))
    assert cvl.local_score(spbn, "B") == cvl.local_score(spbn, "B", spbn.parents("B"))
    assert cvl.local_score(spbn, "C") == cvl.local_score(spbn, "C", spbn.parents("C"))
    assert cvl.local_score(spbn, "D") == cvl.local_score(spbn, "D", spbn.parents("D"))

    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.LinearGaussianCPDType(), "A", []),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "A", []),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.CKDEType(), "B", ["A"]),
        numpy_local_score(pbn.CKDEType(), df, "B", ["A"]),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.LinearGaussianCPDType(), "C", ["A", "B"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df, "C", ["A", "B"]),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.CKDEType(), "D", ["A", "B", "C"]),
        numpy_local_score(pbn.CKDEType(), df, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.CKDEType(), "D", ["A", "B", "C"]),
        numpy_local_score(pbn.CKDEType(), df, "D", ["B", "C", "A"]),
    )


def test_cvl_local_score_null_spbn():
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

    cvl = pbn.CVLikelihood(df_null, 10, seed)

    assert np.isclose(
        cvl.local_score(spbn, "A", []),
        numpy_local_score(pbn.CKDEType(), df_null, "A", []),
    )
    assert np.isclose(
        cvl.local_score(spbn, "B", ["A"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "B", ["A"]),
    )
    assert np.isclose(
        cvl.local_score(spbn, "C", ["A", "B"]),
        numpy_local_score(pbn.CKDEType(), df_null, "C", ["A", "B"]),
    )
    assert np.isclose(
        cvl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        cvl.local_score(spbn, "D", ["A", "B", "C"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "D", ["B", "C", "A"]),
    )

    assert cvl.local_score(spbn, "A") == cvl.local_score(spbn, "A", spbn.parents("A"))
    assert cvl.local_score(spbn, "B") == cvl.local_score(spbn, "B", spbn.parents("B"))
    assert cvl.local_score(spbn, "C") == cvl.local_score(spbn, "C", spbn.parents("C"))
    assert cvl.local_score(spbn, "D") == cvl.local_score(spbn, "D", spbn.parents("D"))

    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.LinearGaussianCPDType(), "A", []),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "A", []),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.CKDEType(), "B", ["A"]),
        numpy_local_score(pbn.CKDEType(), df_null, "B", ["A"]),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.LinearGaussianCPDType(), "C", ["A", "B"]),
        numpy_local_score(pbn.LinearGaussianCPDType(), df_null, "C", ["A", "B"]),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.CKDEType(), "D", ["A", "B", "C"]),
        numpy_local_score(pbn.CKDEType(), df_null, "D", ["A", "B", "C"]),
    )
    assert np.isclose(
        cvl.local_score_node_type(spbn, pbn.CKDEType(), "D", ["A", "B", "C"]),
        numpy_local_score(pbn.CKDEType(), df_null, "D", ["B", "C", "A"]),
    )


def test_cvl_score():
    gbn = pbn.GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    cv = pbn.CVLikelihood(df, 10, 0)

    assert np.isclose(
        cv.score(gbn),
        (
            cv.local_score(gbn, "A", [])
            + cv.local_score(gbn, "B", ["A"])
            + cv.local_score(gbn, "C", ["A", "B"])
            + cv.local_score(gbn, "D", ["A", "B", "C"])
        ),
    )

    spbn = pbn.SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
    )

    assert np.isclose(
        cv.score(spbn),
        (
            cv.local_score(spbn, "A")
            + cv.local_score(spbn, "B")
            + cv.local_score(spbn, "C")
            + cv.local_score(spbn, "D")
        ),
    )
