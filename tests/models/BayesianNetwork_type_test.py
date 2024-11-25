import pybnesian as pbn
from pybnesian import (
    BayesianNetwork,
    BayesianNetworkType,
    ConditionalBayesianNetwork,
    DiscreteBN,
    GaussianNetwork,
    KDENetwork,
    SemiparametricBN,
)
from util_test import generate_normal_data_independent


def test_bn_type():
    g1 = GaussianNetwork(["A", "B", "C", "D"])
    g2 = GaussianNetwork(["A", "B", "C", "D"])
    g3 = GaussianNetwork(["A", "B", "C", "D"])

    assert g1.type() == pbn.GaussianNetworkType()
    assert g1.type() == g2.type()
    assert g1.type() == g3.type()
    assert g2.type() == g3.type()

    s1 = SemiparametricBN(["A", "B", "C", "D"])
    s2 = SemiparametricBN(["A", "B", "C", "D"])
    s3 = SemiparametricBN(["A", "B", "C", "D"])

    assert s1.type() == pbn.SemiparametricBNType()
    assert s1.type() == s2.type()
    assert s1.type() == s3.type()
    assert s2.type() == s3.type()

    k1 = KDENetwork(["A", "B", "C", "D"])
    k2 = KDENetwork(["A", "B", "C", "D"])
    k3 = KDENetwork(["A", "B", "C", "D"])

    assert k1.type() == pbn.KDENetworkType()
    assert k1.type() == k2.type()
    assert k1.type() == k3.type()
    assert k2.type() == k3.type()

    d1 = DiscreteBN(["A", "B", "C", "D"])
    d2 = DiscreteBN(["A", "B", "C", "D"])
    d3 = DiscreteBN(["A", "B", "C", "D"])

    assert d1.type() == pbn.DiscreteBNType()
    assert d1.type() == d2.type()
    assert d1.type() == d3.type()
    assert d2.type() == d3.type()

    assert g1.type() != s1.type()
    assert g1.type() != k1.type()
    assert g1.type() != d1.type()
    assert s1.type() != k1.type()
    assert s1.type() != d1.type()
    assert k1.type() != d1.type()


def test_new_bn_type():
    class MyGaussianNetworkType(BayesianNetworkType):
        def __init__(self):
            BayesianNetworkType.__init__(self)

        def is_homogeneous(self):
            return True

        def can_have_arc(self, model, source, target):
            return source == "A"

    a1 = MyGaussianNetworkType()
    a2 = MyGaussianNetworkType()
    a3 = MyGaussianNetworkType()

    assert a1 == a2
    assert a1 == a3
    assert a2 == a3

    class MySemiparametricBNType(BayesianNetworkType):
        def __init__(self):
            BayesianNetworkType.__init__(self)

    b1 = MySemiparametricBNType()
    b2 = MySemiparametricBNType()
    b3 = MySemiparametricBNType()

    assert b1 == b2
    assert b1 == b3
    assert b2 == b3

    assert a1 != b1

    mybn = BayesianNetwork(a1, ["A", "B", "C", "D"])

    # This type omits the arcs that do not have "A" as source.
    assert mybn.can_add_arc("A", "B")
    assert not mybn.can_add_arc("B", "A")
    assert not mybn.can_add_arc("C", "D")


class MyRestrictedGaussianNetworkType(BayesianNetworkType):
    def __init__(self):
        BayesianNetworkType.__init__(self)

    def is_homogeneous(self):
        return True

    def default_node_type(self):
        return pbn.LinearGaussianCPDType()

    def can_have_arc(self, model, source, target):
        return source == "A"

    def __str__(self):
        return "MyRestrictedGaussianNetworkType"


class SpecificNetwork(BayesianNetwork):
    def __init__(self, variables, arcs=None):
        if arcs is None:
            BayesianNetwork.__init__(self, MyRestrictedGaussianNetworkType(), variables)
        else:
            BayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, arcs
            )


class ConditionalSpecificNetwork(ConditionalBayesianNetwork):
    def __init__(self, variables, interface, arcs=None):
        if arcs is None:
            ConditionalBayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, interface
            )
        else:
            ConditionalBayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, interface, arcs
            )


def test_new_specific_bn_type():
    sp1 = SpecificNetwork(["A", "B", "C", "D"])
    sp2 = SpecificNetwork(["A", "B", "C", "D"], [("A", "B")])
    sp3 = SpecificNetwork(["A", "B", "C", "D"])

    assert sp1.type() == sp2.type()
    assert sp1.type() == sp3.type()
    assert sp2.type() == sp3.type()

    assert sp1.can_add_arc("A", "B")
    assert not sp1.can_add_arc("B", "A")
    assert not sp1.can_add_arc("C", "D")

    assert sp1.num_arcs() == sp3.num_arcs() == 0
    assert sp2.arcs() == [("A", "B")]

    df = generate_normal_data_independent(1000)
    bic = pbn.BIC(df)

    start = SpecificNetwork(["A", "B", "C", "D"])

    hc = pbn.GreedyHillClimbing()
    estimated = hc.estimate(pbn.ArcOperatorSet(), bic, start)
    assert estimated.type() == start.type()
    assert all([s == "A" for s, t in estimated.arcs()])

    # #######################
    # Conditional BN
    # #######################

    csp1 = ConditionalSpecificNetwork(["A", "B"], ["C", "D"])
    csp2 = ConditionalSpecificNetwork(["A", "B"], ["C", "D"], [("A", "B")])
    csp3 = ConditionalSpecificNetwork(["A", "B"], ["C", "D"])

    assert csp1.type() == csp2.type()
    assert csp1.type() == csp3.type()
    assert csp2.type() == csp3.type()

    assert csp1.can_add_arc("A", "B")
    assert not csp1.can_add_arc("B", "A")
    assert not csp1.can_add_arc("C", "D")

    assert csp1.num_arcs() == csp3.num_arcs() == 0
    assert csp2.arcs() == [("A", "B")]

    cstart = ConditionalSpecificNetwork(["A", "C"], ["B", "D"])

    hc = pbn.GreedyHillClimbing()
    cestimated = hc.estimate(pbn.ArcOperatorSet(), bic, cstart)
    assert cestimated.type() == cstart.type()
    assert all([s == "A" for s, t in cestimated.arcs()])
