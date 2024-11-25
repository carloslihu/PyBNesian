import pybnesian as pbn
import pytest


def test_create():
    o = pbn.AddArc("A", "B", 1)
    assert o.source() == "A"
    assert o.target() == "B"
    assert o.delta() == 1

    o = pbn.RemoveArc("A", "B", 2)
    assert o.source() == "A"
    assert o.target() == "B"
    assert o.delta() == 2

    o = pbn.FlipArc("A", "B", 3)
    assert o.source() == "A"
    assert o.target() == "B"
    assert o.delta() == 3

    o = pbn.ChangeNodeType("A", pbn.CKDEType(), 4)
    assert o.node() == "A"
    assert o.node_type() == pbn.CKDEType()
    assert o.delta() == 4


def test_apply():
    gbn = pbn.GaussianNetwork(["A", "B", "C", "D"])
    assert gbn.num_arcs() == 0
    assert not gbn.has_arc("A", "B")

    o = pbn.AddArc("A", "B", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 1
    assert gbn.has_arc("A", "B")

    o = pbn.FlipArc("A", "B", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 1
    assert not gbn.has_arc("A", "B")
    assert gbn.has_arc("B", "A")

    o = pbn.RemoveArc("B", "A", 1)
    o.apply(gbn)
    assert gbn.num_arcs() == 0
    assert not gbn.has_arc("B", "A")

    o = pbn.ChangeNodeType("A", pbn.CKDEType(), 1)
    with pytest.raises(ValueError) as ex:
        o.apply(gbn)
    assert "Wrong factor type" in str(ex.value)

    spbn = pbn.SemiparametricBN(["A", "B", "C", "D"])
    assert spbn.num_arcs() == 0

    o = pbn.ChangeNodeType("A", pbn.CKDEType(), 1)
    assert spbn.node_type("A") == pbn.UnknownFactorType()
    o.apply(spbn)
    assert spbn.node_type("A") == pbn.CKDEType()

    assert not spbn.has_arc("A", "B")
    o = pbn.AddArc("A", "B", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 1
    assert spbn.has_arc("A", "B")

    o = pbn.FlipArc("A", "B", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 1
    assert not spbn.has_arc("A", "B")
    assert spbn.has_arc("B", "A")

    o = pbn.RemoveArc("B", "A", 1)
    o.apply(spbn)
    assert spbn.num_arcs() == 0
    assert not spbn.has_arc("B", "A")


def test_opposite():
    bn = pbn.SemiparametricBN(["A", "B"])
    o = pbn.AddArc("A", "B", 1)
    oppo = o.opposite(bn)
    assert oppo.source() == "A"
    assert oppo.target() == "B"
    assert oppo.delta() == -1
    assert type(oppo) == pbn.RemoveArc

    o = pbn.RemoveArc("A", "B", 1)
    oppo = o.opposite(bn)
    assert oppo.source() == "A"
    assert oppo.target() == "B"
    assert oppo.delta() == -1
    assert type(oppo) == pbn.AddArc

    o = pbn.FlipArc("A", "B", 1)
    oppo = o.opposite(bn)
    assert oppo.source() == "B"
    assert oppo.target() == "A"
    assert oppo.delta() == -1
    assert type(oppo) == pbn.FlipArc

    bn.set_node_type("A", pbn.LinearGaussianCPDType())
    o = pbn.ChangeNodeType("A", pbn.CKDEType(), 1)
    oppo = o.opposite(bn)
    assert oppo.node() == "A"
    assert oppo.node_type() == pbn.LinearGaussianCPDType()
    assert oppo.delta() == -1
    assert type(oppo) == pbn.ChangeNodeType
