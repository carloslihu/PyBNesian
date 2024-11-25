import numpy as np
import pybnesian as pbn
import pytest
from pybnesian import CKDE, LinearGaussianCPD, SemiparametricBN
from util_test import generate_normal_data

df = generate_normal_data(10000)


def test_create_spbn():
    spbn = SemiparametricBN(["A", "B", "C", "D"])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ["A", "B", "C", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == pbn.UnknownFactorType()

    spbn = SemiparametricBN(["A", "B", "C", "D"], [("A", "C")])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 1
    assert spbn.nodes() == ["A", "B", "C", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == pbn.UnknownFactorType()

    spbn = SemiparametricBN([("A", "C"), ("B", "D"), ("C", "D")])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 3
    assert spbn.nodes() == ["A", "C", "B", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == pbn.UnknownFactorType()

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(["A", "B", "C"], [("A", "C", "B")])
    assert "incompatible constructor arguments" in str(ex.value)

    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(["A", "B", "C"], [("A", "D")])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN([("A", "B"), ("B", "C"), ("C", "A")])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(
            ["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "A")]
        )
    assert "must be a DAG" in str(ex.value)

    expected_node_type = {
        "A": pbn.CKDEType(),
        "B": pbn.UnknownFactorType(),
        "C": pbn.CKDEType(),
        "D": pbn.UnknownFactorType(),
    }

    spbn = SemiparametricBN(
        ["A", "B", "C", "D"], [("A", pbn.CKDEType()), ("C", pbn.CKDEType())]
    )
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ["A", "B", "C", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN(
        ["A", "B", "C", "D"],
        [("A", "C")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
    )
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 1
    assert spbn.nodes() == ["A", "B", "C", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    spbn = SemiparametricBN(
        [("A", "C"), ("B", "D"), ("C", "D")],
        [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
    )
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 3
    assert spbn.nodes() == ["A", "C", "B", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == expected_node_type[n]

    with pytest.raises(TypeError) as ex:
        spbn = SemiparametricBN(
            ["A", "B", "C"],
            [("A", "C", "B")],
            [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
        )
    assert "incompatible constructor arguments" in str(ex.value)

    with pytest.raises(IndexError) as ex:
        spbn = SemiparametricBN(
            ["A", "B", "C"],
            [("A", "D")],
            [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
        )
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(
            [("A", "B"), ("B", "C"), ("C", "A")],
            [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
        )
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        spbn = SemiparametricBN(
            ["A", "B", "C", "D"],
            [("A", "B"), ("B", "C"), ("C", "A")],
            [("A", pbn.CKDEType()), ("C", pbn.CKDEType())],
        )
    assert "must be a DAG" in str(ex.value)


def test_node_type():
    spbn = SemiparametricBN(["A", "B", "C", "D"])
    assert spbn.num_nodes() == 4
    assert spbn.num_arcs() == 0
    assert spbn.nodes() == ["A", "B", "C", "D"]

    for n in spbn.nodes():
        assert spbn.node_type(n) == pbn.UnknownFactorType()

    spbn.set_node_type("B", pbn.CKDEType())
    assert spbn.node_type("B") == pbn.CKDEType()
    spbn.set_node_type("B", pbn.LinearGaussianCPDType())
    assert spbn.node_type("B") == pbn.LinearGaussianCPDType()


def test_fit():
    spbn = SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    with pytest.raises(ValueError) as ex:
        for n in spbn.nodes():
            cpd = spbn.cpd(n)
    assert "not added" in str(ex.value)

    spbn.fit(df)

    for n in spbn.nodes():
        cpd = spbn.cpd(n)
        assert cpd.type() == pbn.LinearGaussianCPDType()

        assert type(cpd) == pbn.LinearGaussianCPD
        assert cpd.variable() == n
        assert set(cpd.evidence()) == set(spbn.parents(n))

    spbn.fit(df)

    spbn.remove_arc("A", "B")

    cpd_b = spbn.cpd("B")
    assert type(cpd_b) == pbn.LinearGaussianCPD
    assert cpd_b.evidence != spbn.parents("B")

    spbn.fit(df)
    cpd_b = spbn.cpd("B")
    assert type(cpd_b) == pbn.LinearGaussianCPD
    assert cpd_b.evidence() == spbn.parents("B")

    spbn.set_node_type("C", pbn.CKDEType())

    with pytest.raises(ValueError) as ex:
        cpd_c = spbn.cpd("C")
    assert "not added" in str(ex.value)

    spbn.fit(df)
    cpd_c = spbn.cpd("C")
    assert cpd_c.type() == spbn.node_type("C")


def test_cpd():
    spbn = SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("D", pbn.CKDEType())],
    )

    with pytest.raises(ValueError) as ex:
        spbn.cpd("A")
    assert "not added" in str(ex.value)

    spbn.fit(df)

    assert spbn.cpd("A").type() == pbn.LinearGaussianCPDType()
    assert spbn.cpd("B").type() == pbn.LinearGaussianCPDType()
    assert spbn.cpd("C").type() == pbn.LinearGaussianCPDType()
    assert spbn.cpd("D").type() == pbn.CKDEType()

    assert spbn.cpd("A").fitted()
    assert spbn.cpd("B").fitted()
    assert spbn.cpd("C").fitted()
    assert spbn.cpd("D").fitted()


def test_add_cpds():
    spbn = SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
        [("D", pbn.CKDEType())],
    )

    assert spbn.node_type("A") == pbn.UnknownFactorType()
    spbn.add_cpds([CKDE("A", [])])
    assert spbn.node_type("A") == pbn.CKDEType()

    with pytest.raises(ValueError) as ex:
        spbn.add_cpds([LinearGaussianCPD("D", ["A", "B", "C"])])
    assert "Bayesian network expects type" in str(ex.value)

    lg = LinearGaussianCPD("B", ["A"], [2.5, 1.65], 4)
    ckde = CKDE("D", ["A", "B", "C"])
    assert lg.fitted()
    assert not ckde.fitted()

    spbn.add_cpds([lg, ckde])

    spbn.set_node_type("A", pbn.UnknownFactorType())
    with pytest.raises(ValueError) as ex:
        spbn.cpd("A").fitted()
    assert (
        'CPD of variable "A" not added. Call add_cpds() or fit() to add the CPD.'
        in str(ex.value)
    )

    assert spbn.cpd("B").fitted()

    with pytest.raises(ValueError) as ex:
        spbn.cpd("C").fitted()
    assert (
        'CPD of variable "C" not added. Call add_cpds() or fit() to add the CPD.'
        in str(ex.value)
    )

    assert not spbn.cpd("D").fitted()


def test_logl():
    spbn = SemiparametricBN(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    spbn.fit(df)

    test_df = generate_normal_data(5000)
    ll = spbn.logl(test_df)
    sll = spbn.slogl(test_df)

    sum_ll = np.zeros((5000,))
    sum_sll = 0

    for n in spbn.nodes():
        cpd = spbn.cpd(n)
        log_likelihood = cpd.logl(test_df)
        sum_log_likelihood = cpd.slogl(test_df)
        assert np.all(np.isclose(sum_log_likelihood, log_likelihood.sum()))
        sum_ll += log_likelihood
        sum_sll += sum_log_likelihood

    assert np.all(np.isclose(ll, sum_ll))
    assert np.isclose(sll, ll.sum())
    assert sll == sum_sll
