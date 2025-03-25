import numpy as np
import pybnesian as pbn
import pytest
from helpers.data import DATA_SIZE, generate_normal_data

df = generate_normal_data(DATA_SIZE)


def test_create_change_node():
    gbn = pbn.GaussianNetwork(["A", "B", "C", "D"])

    cv = pbn.CVLikelihood(df)

    node_op = pbn.ChangeNodeTypeSet()

    with pytest.raises(ValueError) as ex:
        node_op.cache_scores(gbn, cv)
    assert "can only be used with non-homogeneous" in str(ex.value)


def test_lists():
    gbn = pbn.GaussianNetwork(["A", "B", "C", "D"])
    bic = pbn.BIC(df)
    arc_op = pbn.ArcOperatorSet()

    arc_op.set_arc_blacklist([("B", "A")])
    arc_op.set_arc_whitelist([("B", "C")])
    arc_op.set_max_indegree(3)
    arc_op.set_type_whitelist([("A", pbn.LinearGaussianCPDType())])

    arc_op.cache_scores(gbn, bic)

    arc_op.set_arc_blacklist([("E", "A")])

    with pytest.raises(ValueError) as ex:
        arc_op.cache_scores(gbn, bic)
    assert "not present in the graph" in str(ex.value)

    arc_op.set_arc_whitelist([("E", "A")])

    with pytest.raises(ValueError) as ex:
        arc_op.cache_scores(gbn, bic)
    assert "not present in the graph" in str(ex.value)


def test_check_max_score():
    gbn = pbn.GaussianNetwork(["C", "D"])

    bic = pbn.BIC(df)
    arc_op = pbn.ArcOperatorSet()

    arc_op.cache_scores(gbn, bic)
    op = arc_op.find_max(gbn)

    assert np.isclose(
        op.delta(), (bic.local_score(gbn, "D", ["C"]) - bic.local_score(gbn, "D"))
    )

    # BIC is decomposable so the best operation is the arc in reverse direction.
    arc_op.set_arc_blacklist([(op.source(), op.target())])
    arc_op.cache_scores(gbn, bic)

    op2 = arc_op.find_max(gbn)

    assert op.source() == op2.target()
    assert op.target() == op2.source()
    assert (type(op) == type(op2)) and (type(op) == pbn.AddArc)


def test_nomax():
    gbn = pbn.GaussianNetwork(["A", "B"])

    bic = pbn.BIC(df)
    arc_op = pbn.ArcOperatorSet(whitelist=[("A", "B")])
    arc_op.cache_scores(gbn, bic)

    op = arc_op.find_max(gbn)

    assert op is None
