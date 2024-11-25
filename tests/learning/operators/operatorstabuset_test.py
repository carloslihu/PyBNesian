import pybnesian as pbn


def test_OperatorTabuSet():
    tabu_set = pbn.OperatorTabuSet()

    assert tabu_set.empty()

    assert not tabu_set.contains(pbn.AddArc("A", "B", 1))
    tabu_set.insert(pbn.AddArc("A", "B", 2))
    assert not tabu_set.empty()
    assert tabu_set.contains(pbn.AddArc("A", "B", 3))

    assert not tabu_set.contains(pbn.RemoveArc("B", "C", 4))
    tabu_set.insert(pbn.RemoveArc("B", "C", 5))
    assert tabu_set.contains(pbn.RemoveArc("B", "C", 6))

    assert not tabu_set.contains(pbn.FlipArc("C", "D", 7))
    tabu_set.insert(pbn.RemoveArc("C", "D", 8))
    assert tabu_set.contains(pbn.RemoveArc("C", "D", 9))

    tabu_set.clear()
    assert tabu_set.empty()
