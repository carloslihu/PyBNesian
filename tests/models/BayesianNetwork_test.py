import numpy as np
import pybnesian as pbn
import pytest
from pybnesian import BayesianNetwork, GaussianNetwork
from util_test import generate_normal_data

df = generate_normal_data(10000)


def test_create_bn():
    gbn = GaussianNetwork(["A", "B", "C", "D"])

    assert gbn.num_nodes() == 4
    assert gbn.num_arcs() == 0
    assert gbn.nodes() == ["A", "B", "C", "D"]

    gbn = GaussianNetwork(["A", "B", "C", "D"], [("A", "C")])
    assert gbn.num_nodes() == 4
    assert gbn.num_arcs() == 1
    assert gbn.nodes() == ["A", "B", "C", "D"]

    gbn = GaussianNetwork([("A", "C"), ("B", "D"), ("C", "D")])
    assert gbn.num_nodes() == 4
    assert gbn.num_arcs() == 3
    assert gbn.nodes() == ["A", "C", "B", "D"]

    with pytest.raises(TypeError) as ex:
        gbn = GaussianNetwork(["A", "B", "C"], [("A", "C", "B")])
    assert "incompatible constructor arguments" in str(ex.value)

    with pytest.raises(IndexError) as ex:
        gbn = GaussianNetwork(["A", "B", "C"], [("A", "D")])
    assert "not present in the graph" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = GaussianNetwork([("A", "B"), ("B", "C"), ("C", "A")])
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = GaussianNetwork(
            ["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "A")]
        )
    assert "must be a DAG" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn = BayesianNetwork(
            pbn.GaussianNetworkType(), ["A", "B", "C", "D"], [], [("A", pbn.CKDEType())]
        )
    assert "Wrong factor type" in str(ex.value)


def gbn_generator():
    # Test different Networks created with different constructors.
    gbn = GaussianNetwork(["A", "B", "C", "D"])
    yield gbn
    gbn = GaussianNetwork([("A", "C"), ("B", "D"), ("C", "D")])
    yield gbn
    gbn = GaussianNetwork(["A", "B", "C", "D"], [("A", "B"), ("B", "C")])
    yield gbn


def test_nodes_util():
    for gbn in gbn_generator():
        assert gbn.num_nodes() == 4

        nodes = gbn.nodes()
        indices = gbn.indices()

        assert nodes[gbn.index("A")] == "A"
        assert nodes[gbn.index("B")] == "B"
        assert nodes[gbn.index("C")] == "C"
        assert nodes[gbn.index("D")] == "D"

        assert indices[gbn.name(0)] == 0
        assert indices[gbn.name(1)] == 1
        assert indices[gbn.name(2)] == 2
        assert indices[gbn.name(3)] == 3

        assert gbn.contains_node("A")
        assert gbn.contains_node("B")
        assert gbn.contains_node("C")
        assert gbn.contains_node("D")
        assert not gbn.contains_node("E")


def test_parent_children():
    gbn = GaussianNetwork(["A", "B", "C", "D"])

    assert gbn.num_parents("A") == 0
    assert gbn.num_parents("B") == 0
    assert gbn.num_parents("C") == 0
    assert gbn.num_parents("D") == 0

    assert gbn.parents("A") == []
    assert gbn.parents("B") == []
    assert gbn.parents("C") == []
    assert gbn.parents("D") == []

    assert gbn.num_children("A") == 0
    assert gbn.num_children("B") == 0
    assert gbn.num_children("C") == 0
    assert gbn.num_children("D") == 0

    gbn = GaussianNetwork([("A", "C"), ("B", "D"), ("C", "D")])

    assert gbn.num_parents("A") == 0
    assert gbn.num_parents("B") == 0
    assert gbn.num_parents("C") == 1
    assert gbn.num_parents("D") == 2

    assert gbn.parents("A") == []
    assert gbn.parents("B") == []
    assert gbn.parents("C") == ["A"]
    assert set(gbn.parents("D")) == set(["B", "C"])

    assert gbn.num_children("A") == 1
    assert gbn.num_children("B") == 1
    assert gbn.num_children("C") == 1
    assert gbn.num_children("D") == 0

    gbn = GaussianNetwork(["A", "B", "C", "D"], [("A", "B"), ("B", "C")])

    assert gbn.num_parents("A") == 0
    assert gbn.num_parents("B") == 1
    assert gbn.num_parents("C") == 1
    assert gbn.num_parents("D") == 0

    assert gbn.parents("A") == []
    assert gbn.parents("B") == ["A"]
    assert gbn.parents("C") == ["B"]
    assert gbn.parents("D") == []

    assert gbn.num_children("A") == 1
    assert gbn.num_children("B") == 1
    assert gbn.num_children("C") == 0
    assert gbn.num_children("D") == 0


def test_arcs():
    gbn = GaussianNetwork(["A", "B", "C", "D"])

    assert gbn.num_arcs() == 0
    assert gbn.arcs() == []
    assert not gbn.has_arc("A", "B")

    gbn.add_arc("A", "B")
    assert gbn.num_arcs() == 1
    assert gbn.arcs() == [("A", "B")]
    assert gbn.parents("B") == ["A"]
    assert gbn.num_parents("B") == 1
    assert gbn.num_children("A") == 1
    assert gbn.has_arc("A", "B")

    gbn.add_arc("B", "C")
    assert gbn.num_arcs() == 2
    assert set(gbn.arcs()) == set([("A", "B"), ("B", "C")])
    assert gbn.parents("C") == ["B"]
    assert gbn.num_parents("C") == 1
    assert gbn.num_children("B") == 1
    assert gbn.has_arc("B", "C")

    gbn.add_arc("D", "C")
    assert gbn.num_arcs() == 3
    assert set(gbn.arcs()) == set([("A", "B"), ("B", "C"), ("D", "C")])
    assert set(gbn.parents("C")) == set(["B", "D"])
    assert gbn.num_parents("C") == 2
    assert gbn.num_children("D") == 1
    assert gbn.has_arc("D", "C")

    assert gbn.has_path("A", "C")
    assert not gbn.has_path("A", "D")
    assert gbn.has_path("B", "C")
    assert gbn.has_path("D", "C")

    assert not gbn.can_add_arc("C", "A")
    # This edge exists, but virtually we consider that the addition is allowed.
    assert gbn.can_add_arc("B", "C")
    assert gbn.can_add_arc("D", "A")

    gbn.add_arc("B", "D")
    assert gbn.num_arcs() == 4
    assert set(gbn.arcs()) == set([("A", "B"), ("B", "C"), ("D", "C"), ("B", "D")])
    assert gbn.parents("D") == ["B"]
    assert gbn.num_parents("D") == 1
    assert gbn.num_children("B") == 2
    assert gbn.has_arc("B", "D")

    assert gbn.has_path("A", "D")
    assert not gbn.can_add_arc("D", "A")
    assert not gbn.can_flip_arc("B", "C")
    assert gbn.can_flip_arc("A", "B")
    # This edge does not exist, but it could be flipped if it did.
    assert gbn.can_flip_arc("D", "A")

    # We can add an edge twice without changes.
    gbn.add_arc("B", "D")
    assert gbn.num_arcs() == 4
    assert set(gbn.arcs()) == set([("A", "B"), ("B", "C"), ("D", "C"), ("B", "D")])
    assert gbn.parents("D") == ["B"]
    assert gbn.num_parents("D") == 1
    assert gbn.num_children("B") == 2
    assert gbn.has_arc("B", "D")

    gbn.remove_arc("B", "C")
    assert gbn.num_arcs() == 3
    assert set(gbn.arcs()) == set([("A", "B"), ("D", "C"), ("B", "D")])
    assert gbn.parents("C") == ["D"]
    assert gbn.num_parents("C") == 1
    assert gbn.num_children("B") == 1
    assert not gbn.has_arc("B", "C")

    assert gbn.can_add_arc("B", "C")
    assert not gbn.can_add_arc("C", "B")
    assert gbn.has_path("A", "C")
    assert gbn.has_path("B", "C")

    gbn.remove_arc("D", "C")
    assert gbn.num_arcs() == 2
    assert set(gbn.arcs()) == set([("A", "B"), ("B", "D")])
    assert gbn.parents("C") == []
    assert gbn.num_parents("C") == 0
    assert gbn.num_children("D") == 0
    assert not gbn.has_arc("D", "C")

    assert gbn.can_add_arc("B", "C")
    assert gbn.can_add_arc("C", "B")
    assert not gbn.has_path("A", "C")
    assert not gbn.has_path("B", "C")


def test_bn_fit():
    gbn = GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    with pytest.raises(ValueError) as ex:
        for n in gbn.nodes():
            cpd = gbn.cpd(n)
    assert "not added" in str(ex.value)

    gbn.fit(df)

    for n in gbn.nodes():
        cpd = gbn.cpd(n)
        assert cpd.variable() == n
        assert cpd.evidence() == gbn.parents(n)

    gbn.fit(df)

    gbn.remove_arc("A", "B")

    cpd_b = gbn.cpd("B")
    assert cpd_b.evidence != gbn.parents("B")

    gbn.fit(df)

    cpd_b = gbn.cpd("B")
    assert cpd_b.evidence() == gbn.parents("B")


def test_add_cpds():
    gbn = GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD("E", [])])
    assert "variable which is not present" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD("A", ["E"])])
    assert "Evidence variable" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD("A", ["B"])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD("B", [])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD("B", ["C"])])
    assert "CPD do not have the model's parent set as evidence" in str(ex.value)

    lg = pbn.LinearGaussianCPD("B", ["A"], [2.5, 1.65], 4)
    assert lg.fitted()

    gbn.add_cpds([lg])

    cpd_b = gbn.cpd("B")
    assert cpd_b.variable() == "B"
    assert cpd_b.evidence() == ["A"]
    assert cpd_b.fitted()
    assert np.all(cpd_b.beta == np.asarray([2.5, 1.65]))
    assert cpd_b.variance == 4

    with pytest.raises(ValueError) as ex:
        gbn.cpd("A")
    assert (
        'CPD of variable "A" not added. Call add_cpds() or fit() to add the CPD.'
        in str(ex.value)
    )

    with pytest.raises(ValueError) as ex:
        gbn.cpd("C")
    assert (
        'CPD of variable "C" not added. Call add_cpds() or fit() to add the CPD.'
        in str(ex.value)
    )

    with pytest.raises(ValueError) as ex:
        gbn.cpd("D")
    assert (
        'CPD of variable "D" not added. Call add_cpds() or fit() to add the CPD.'
        in str(ex.value)
    )

    with pytest.raises(ValueError) as ex:
        gbn.add_cpds([pbn.LinearGaussianCPD("E", [])])
    assert "variable which is not present" in str(ex.value)


def test_bn_logl():
    gbn = GaussianNetwork(
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")]
    )

    gbn.fit(df)

    test_df = generate_normal_data(5000)
    ll = gbn.logl(test_df)
    sll = gbn.slogl(test_df)

    sum_ll = np.zeros((5000,))
    sum_sll = 0

    for n in gbn.nodes():
        cpd = gbn.cpd(n)
        log_likelihood = cpd.logl(test_df)
        sum_log_likelihood = cpd.slogl(test_df)
        assert np.all(np.isclose(sum_log_likelihood, log_likelihood.sum()))
        sum_ll += log_likelihood
        sum_sll += sum_log_likelihood

    assert np.all(np.isclose(ll, sum_ll))
    assert np.isclose(sll, ll.sum())
    assert sll == sum_sll


def test_bn_sample():
    gbn = GaussianNetwork(
        ["A", "C", "B", "D"],
        [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
    )

    gbn.fit(df)
    sample = gbn.sample(1000, 0, False)

    # Not ordered, so topological sort.
    assert sample.schema.names == ["A", "B", "C", "D"]
    assert sample.num_rows == 1000

    sample_ordered = gbn.sample(1000, 0, True)
    assert sample_ordered.schema.names == ["A", "C", "B", "D"]
    assert sample_ordered.num_rows == 1000

    assert sample.column(0).equals(sample_ordered.column(0))
    assert sample.column(1).equals(sample_ordered.column(2))
    assert sample.column(2).equals(sample_ordered.column(1))
    assert sample.column(3).equals(sample_ordered.column(3))

    other_seed = gbn.sample(1000, 1, False)

    assert not sample.column(0).equals(other_seed.column(0))
    assert not sample.column(1).equals(other_seed.column(2))
    assert not sample.column(2).equals(other_seed.column(1))
    assert not sample.column(3).equals(other_seed.column(3))
