import pickle

import pyarrow as pa
import pybnesian as pbn
import pytest
from helpers.data import generate_discrete_data, generate_normal_data_independent


@pytest.fixture
def gaussian_bytes():
    gaussian = pbn.GaussianNetwork(["A", "B", "C", "D"], [("A", "B")])
    return pickle.dumps(gaussian)


@pytest.fixture
def spbn_bytes():
    spbn = pbn.SemiparametricBN(
        ["A", "B", "C", "D"], [("A", "B")], [("B", pbn.CKDEType())]
    )
    return pickle.dumps(spbn)


@pytest.fixture
def kde_bytes():
    kde = pbn.KDENetwork(["A", "B", "C", "D"], [("A", "B")])
    return pickle.dumps(kde)


@pytest.fixture
def discrete_bytes():
    discrete = pbn.DiscreteBN(["A", "B", "C", "D"], [("A", "B")])
    return pickle.dumps(discrete)


class MyRestrictedGaussianNetworkType(pbn.BayesianNetworkType):
    def __init__(self):
        pbn.BayesianNetworkType.__init__(self)

    def is_homogeneous(self):
        return True

    def default_node_type(self):
        return pbn.LinearGaussianCPDType()

    def can_have_arc(self, model, source, target):
        return "A" in source

    def new_bn(self, nodes):
        return NewBN(nodes)

    def new_cbn(self, nodes, interface_nodes):
        return ConditionalNewBN(nodes, interface_nodes)

    def __str__(self):
        return "MyRestrictedGaussianNetworkType"


@pytest.fixture
def genericbn_bytes():
    gen = pbn.BayesianNetwork(
        MyRestrictedGaussianNetworkType(), ["A", "B", "C", "D"], [("A", "B")]
    )
    return pickle.dumps(gen)


class NewBN(pbn.BayesianNetwork):
    def __init__(self, variables, arcs=None):
        if arcs is None:
            pbn.BayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables
            )
        else:
            pbn.BayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, arcs
            )


@pytest.fixture
def newbn_bytes():
    new = NewBN(["A", "B", "C", "D"], [("A", "B")])
    return pickle.dumps(new)


class NonHomogeneousType(pbn.BayesianNetworkType):
    def __init__(self):
        pbn.BayesianNetworkType.__init__(self)

    def is_homogeneous(self):
        return False

    def data_default_node_type(self, dt):
        if dt.equals(pa.float64()) or dt.equals(pa.float32()):
            return [pbn.LinearGaussianCPDType()]
        else:
            raise ValueError("Data type not compatible with NonHomogeneousType")

    def new_bn(self, nodes):
        return OtherBN(nodes)

    def new_cbn(self, nodes, interface_nodes):
        return ConditionalOtherBN(nodes, interface_nodes)

    def __str__(self):
        return "NonHomogeneousType"


class OtherBN(pbn.BayesianNetwork):
    def __init__(self, variables, arcs=None, node_types=None):
        if arcs is None:
            if node_types is None:
                pbn.BayesianNetwork.__init__(self, NonHomogeneousType(), variables)
            else:
                pbn.BayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, node_types
                )
        else:
            if node_types is None:
                pbn.BayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, arcs
                )
            else:
                pbn.BayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, arcs, node_types
                )

        self.extra_info = "extra"

    def __getstate_extra__(self):
        return self.extra_info

    def __setstate_extra__(self, t):
        self.extra_info = t


@pytest.fixture
def otherbn_bytes():
    other = OtherBN(
        ["A", "B", "C", "D"],
        [("A", "B")],
        [
            ("B", pbn.LinearGaussianCPDType()),
            ("C", pbn.CKDEType()),
            ("D", pbn.DiscreteFactorType()),
        ],
    )
    return pickle.dumps(other)


def test_serialization_bn_model(
    gaussian_bytes,
    spbn_bytes,
    kde_bytes,
    discrete_bytes,
    genericbn_bytes,
    newbn_bytes,
    otherbn_bytes,
):
    loaded_g = pickle.loads(gaussian_bytes)
    assert set(loaded_g.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_g.arcs() == [("A", "B")]
    assert loaded_g.type() == pbn.GaussianNetworkType()

    loaded_s = pickle.loads(spbn_bytes)
    assert set(loaded_s.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_s.arcs() == [("A", "B")]
    assert loaded_s.type() == pbn.SemiparametricBNType()
    assert loaded_s.node_types() == {
        "A": pbn.UnknownFactorType(),
        "B": pbn.CKDEType(),
        "C": pbn.UnknownFactorType(),
        "D": pbn.UnknownFactorType(),
    }

    loaded_k = pickle.loads(kde_bytes)
    assert set(loaded_k.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_k.arcs() == [("A", "B")]
    assert loaded_k.type() == pbn.KDENetworkType()

    loaded_d = pickle.loads(discrete_bytes)
    assert set(loaded_d.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_d.arcs() == [("A", "B")]
    assert loaded_d.type() == pbn.DiscreteBNType()

    loaded_gen = pickle.loads(genericbn_bytes)
    assert set(loaded_gen.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_gen.arcs() == [("A", "B")]
    assert loaded_gen.type() == MyRestrictedGaussianNetworkType()

    loaded_nn = pickle.loads(newbn_bytes)
    assert set(loaded_g.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_nn.arcs() == [("A", "B")]
    assert loaded_nn.type() == MyRestrictedGaussianNetworkType()

    loaded_o = pickle.loads(otherbn_bytes)
    assert set(loaded_g.nodes()) == set(["A", "B", "C", "D"])
    assert loaded_o.arcs() == [("A", "B")]
    assert loaded_o.type() == NonHomogeneousType()
    assert loaded_o.node_types() == {
        "A": pbn.UnknownFactorType(),
        "B": pbn.LinearGaussianCPDType(),
        "C": pbn.CKDEType(),
        "D": pbn.DiscreteFactorType(),
    }
    assert loaded_o.extra_info == "extra"

    assert loaded_nn.type() != loaded_o.type()


@pytest.fixture
def gaussian_partial_fit_bytes():
    gaussian = pbn.GaussianNetwork(["A", "B", "C", "D"], [("A", "B")])
    lg = pbn.pbn.LinearGaussianCPD("B", ["A"], [1, 2], 2)
    gaussian.add_cpds([lg])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)


@pytest.fixture
def gaussian_fit_bytes():
    gaussian = pbn.GaussianNetwork(["A", "B", "C", "D"], [("A", "B")])
    lg_a = pbn.LinearGaussianCPD("A", [], [0], 0.5)
    lg_b = pbn.LinearGaussianCPD("B", ["A"], [1, 2], 2)
    lg_c = pbn.LinearGaussianCPD("C", [], [2], 1)
    lg_d = pbn.LinearGaussianCPD("D", [], [3], 1.5)
    gaussian.add_cpds([lg_a, lg_b, lg_c, lg_d])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)


@pytest.fixture
def other_partial_fit_bytes():
    other = OtherBN(
        ["A", "B", "C", "D"],
        [("A", "B")],
        [
            ("B", pbn.LinearGaussianCPDType()),
            ("C", pbn.CKDEType()),
            ("D", pbn.DiscreteFactorType()),
        ],
    )
    lg = pbn.LinearGaussianCPD("B", ["A"], [1, 2], 2)
    other.add_cpds([lg])
    other.include_cpd = True
    return pickle.dumps(other)


@pytest.fixture
def other_fit_bytes():
    other = OtherBN(
        ["A", "B", "C", "D"],
        [("A", "B")],
        [
            ("B", pbn.LinearGaussianCPDType()),
            ("C", pbn.CKDEType()),
            ("D", pbn.DiscreteFactorType()),
        ],
    )
    cpd_a = pbn.LinearGaussianCPD("A", [], [0], 0.5)
    cpd_b = pbn.LinearGaussianCPD("B", ["A"], [1, 2], 2)

    df_continuous = generate_normal_data_independent(100)
    cpd_c = pbn.CKDE("C", [])
    cpd_c.fit(df_continuous)

    df_discrete = generate_discrete_data(100)
    cpd_d = pbn.DiscreteFactor("D", [])
    cpd_d.fit(df_discrete)

    other.add_cpds([cpd_a, cpd_b, cpd_c, cpd_d])

    other.include_cpd = True
    return pickle.dumps(other)


def test_serialization_fitted_bn(
    gaussian_partial_fit_bytes,
    gaussian_fit_bytes,
    other_partial_fit_bytes,
    other_fit_bytes,
):
    # ####################
    # Gaussian partial fit
    # ####################
    loaded_partial = pickle.loads(gaussian_partial_fit_bytes)
    assert not loaded_partial.fitted()
    cpd = loaded_partial.cpd("B")
    assert cpd.variable() == "B"
    assert cpd.evidence() == ["A"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    # ####################
    # Gaussian fit
    # ####################
    loaded_fitted = pickle.loads(gaussian_fit_bytes)
    assert loaded_fitted.fitted()

    cpd_a = loaded_fitted.cpd("A")
    assert cpd_a.variable() == "A"
    assert cpd_a.evidence() == []
    assert cpd_a.beta == [0]
    assert cpd_a.variance == 0.5

    cpd_b = loaded_fitted.cpd("B")
    assert cpd_b.variable() == "B"
    assert cpd_b.evidence() == ["A"]
    assert list(cpd_b.beta) == [1, 2]
    assert cpd_b.variance == 2

    cpd_c = loaded_fitted.cpd("C")
    assert cpd_c.variable() == "C"
    assert cpd_c.evidence() == []
    assert cpd_c.beta == [2]
    assert cpd_c.variance == 1

    cpd_d = loaded_fitted.cpd("D")
    assert cpd_d.variable() == "D"
    assert cpd_d.evidence() == []
    assert cpd_d.beta == [3]
    assert cpd_d.variance == 1.5

    # ####################
    # OtherBN homogeneous partial fit
    # ####################
    loaded_other = pickle.loads(other_partial_fit_bytes)
    assert not loaded_other.fitted()
    cpd = loaded_partial.cpd("B")
    assert cpd.variable() == "B"
    assert cpd.evidence() == ["A"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    # ####################
    # OtherBN homogeneous fit
    # ####################
    loaded_other_fitted = pickle.loads(other_fit_bytes)
    assert loaded_other_fitted.fitted()

    cpd_a = loaded_other_fitted.cpd("A")
    assert cpd_a.variable() == "A"
    assert cpd_a.evidence() == []
    assert cpd_a.beta == [0]
    assert cpd_a.variance == 0.5
    assert cpd_a.type() == pbn.LinearGaussianCPDType()

    cpd_b = loaded_other_fitted.cpd("B")
    assert cpd_b.variable() == "B"
    assert cpd_b.evidence() == ["A"]
    assert list(cpd_b.beta) == [1, 2]
    assert cpd_b.variance == 2
    assert cpd_b.type() == pbn.LinearGaussianCPDType()

    cpd_c = loaded_other_fitted.cpd("C")
    assert cpd_c.variable() == "C"
    assert cpd_c.evidence() == []
    assert cpd_c.fitted()
    assert cpd_c.num_instances() == 100
    assert cpd_c.type() == pbn.CKDEType()

    cpd_d = loaded_other_fitted.cpd("D")
    assert cpd_d.variable() == "D"
    assert cpd_d.evidence() == []
    assert cpd_d.fitted()
    assert cpd_d.type() == pbn.DiscreteFactorType()


# ##########################
# Conditional BN
# ##########################


@pytest.fixture
def cond_gaussian_bytes():
    gaussian = pbn.ConditionalGaussianNetwork(["C", "D"], ["A", "B"], [("A", "C")])
    return pickle.dumps(gaussian)


@pytest.fixture
def cond_spbn_bytes():
    spbn = pbn.ConditionalSemiparametricBN(
        ["C", "D"], ["A", "B"], [("A", "C")], [("C", pbn.CKDEType())]
    )
    return pickle.dumps(spbn)


@pytest.fixture
def cond_kde_bytes():
    kde = pbn.ConditionalKDENetwork(["C", "D"], ["A", "B"], [("A", "C")])
    return pickle.dumps(kde)


@pytest.fixture
def cond_discrete_bytes():
    discrete = pbn.ConditionalDiscreteBN(["C", "D"], ["A", "B"], [("A", "C")])
    return pickle.dumps(discrete)


@pytest.fixture
def cond_genericbn_bytes():
    gen = pbn.ConditionalBayesianNetwork(
        MyRestrictedGaussianNetworkType(), ["C", "D"], ["A", "B"], [("A", "C")]
    )
    return pickle.dumps(gen)


class ConditionalNewBN(pbn.ConditionalBayesianNetwork):
    def __init__(self, variables, interface, arcs=None):
        if arcs is None:
            pbn.ConditionalBayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, interface
            )
        else:
            pbn.ConditionalBayesianNetwork.__init__(
                self, MyRestrictedGaussianNetworkType(), variables, interface, arcs
            )


@pytest.fixture
def cond_newbn_bytes():
    new = ConditionalNewBN(["C", "D"], ["A", "B"], [("A", "C")])
    return pickle.dumps(new)


class ConditionalOtherBN(pbn.ConditionalBayesianNetwork):
    def __init__(self, variables, interface, arcs=None, node_types=None):
        if arcs is None:
            if node_types is None:
                pbn.ConditionalBayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, interface
                )
            else:
                pbn.ConditionalBayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, interface, node_types
                )
        else:
            if node_types is None:
                pbn.ConditionalBayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, interface, arcs
                )
            else:
                pbn.ConditionalBayesianNetwork.__init__(
                    self, NonHomogeneousType(), variables, interface, arcs, node_types
                )

        self.extra_info = "extra"

    def __getstate_extra__(self):
        return self.extra_info

    def __setstate_extra__(self, t):
        self.extra_info = t


@pytest.fixture
def cond_otherbn_bytes():
    other = ConditionalOtherBN(
        ["C", "D"],
        ["A", "B"],
        [("A", "C")],
        [
            ("B", pbn.LinearGaussianCPDType()),
            ("C", pbn.CKDEType()),
            ("D", pbn.DiscreteFactorType()),
        ],
    )
    return pickle.dumps(other)


def test_serialization_conditional_bn_model(
    cond_gaussian_bytes,
    cond_spbn_bytes,
    cond_kde_bytes,
    cond_discrete_bytes,
    cond_genericbn_bytes,
    cond_newbn_bytes,
    cond_otherbn_bytes,
    newbn_bytes,
    otherbn_bytes,
):
    loaded_g = pickle.loads(cond_gaussian_bytes)
    assert set(loaded_g.nodes()) == set(["C", "D"])
    assert set(loaded_g.interface_nodes()) == set(["A", "B"])
    assert loaded_g.arcs() == [("A", "C")]
    assert loaded_g.type() == pbn.GaussianNetworkType()

    loaded_s = pickle.loads(cond_spbn_bytes)
    assert set(loaded_s.nodes()) == set(["C", "D"])
    assert set(loaded_s.interface_nodes()) == set(["A", "B"])
    assert loaded_s.arcs() == [("A", "C")]
    assert loaded_s.type() == pbn.SemiparametricBNType()
    assert loaded_s.node_types() == {"C": pbn.CKDEType(), "D": pbn.UnknownFactorType()}

    loaded_k = pickle.loads(cond_kde_bytes)
    assert set(loaded_k.nodes()) == set(["C", "D"])
    assert set(loaded_k.interface_nodes()) == set(["A", "B"])
    assert loaded_k.arcs() == [("A", "C")]
    assert loaded_k.type() == pbn.KDENetworkType()

    loaded_d = pickle.loads(cond_discrete_bytes)
    assert set(loaded_d.nodes()) == set(["C", "D"])
    assert set(loaded_d.interface_nodes()) == set(["A", "B"])
    assert loaded_d.arcs() == [("A", "C")]
    assert loaded_d.type() == pbn.DiscreteBNType()

    loaded_gen = pickle.loads(cond_genericbn_bytes)
    assert set(loaded_gen.nodes()) == set(["C", "D"])
    assert set(loaded_gen.interface_nodes()) == set(["A", "B"])
    assert loaded_gen.arcs() == [("A", "C")]
    assert loaded_gen.type() == MyRestrictedGaussianNetworkType()

    loaded_nn = pickle.loads(cond_newbn_bytes)
    assert set(loaded_nn.nodes()) == set(["C", "D"])
    assert set(loaded_nn.interface_nodes()) == set(["A", "B"])
    assert loaded_nn.arcs() == [("A", "C")]
    assert loaded_nn.type() == MyRestrictedGaussianNetworkType()

    loaded_o = pickle.loads(cond_otherbn_bytes)
    assert set(loaded_o.nodes()) == set(["C", "D"])
    assert set(loaded_o.interface_nodes()) == set(["A", "B"])
    assert loaded_o.arcs() == [("A", "C")]
    assert loaded_o.type() == NonHomogeneousType()
    assert loaded_o.node_types() == {"C": pbn.CKDEType(), "D": pbn.DiscreteFactorType()}
    assert loaded_o.extra_info == "extra"

    assert loaded_nn.type() != loaded_o.type()

    loaded_unconditional_nn = pickle.loads(newbn_bytes)
    loaded_unconditional_o = pickle.loads(otherbn_bytes)

    assert loaded_nn.type() == loaded_unconditional_nn.type()
    assert loaded_o.type() == loaded_unconditional_o.type()


@pytest.fixture
def cond_gaussian_partial_fit_bytes():
    gaussian = pbn.ConditionalGaussianNetwork(["C", "D"], ["A", "B"], [("A", "C")])
    lg = pbn.LinearGaussianCPD("C", ["A"], [1, 2], 2)
    gaussian.add_cpds([lg])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)


@pytest.fixture
def cond_gaussian_fit_bytes():
    gaussian = pbn.ConditionalGaussianNetwork(["C", "D"], ["A", "B"], [("A", "C")])
    lg_c = pbn.LinearGaussianCPD("C", ["A"], [1, 2], 2)
    lg_d = pbn.LinearGaussianCPD("D", [], [3], 1.5)
    gaussian.add_cpds([lg_c, lg_d])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)


@pytest.fixture
def cond_other_partial_fit_bytes():
    other = ConditionalOtherBN(
        ["C", "D"],
        ["A", "B"],
        [("A", "C")],
        [("C", pbn.CKDEType()), ("D", pbn.LinearGaussianCPDType())],
    )
    lg = pbn.LinearGaussianCPD("D", [], [3], 1.5)
    other.add_cpds([lg])
    other.include_cpd = True
    return pickle.dumps(other)


@pytest.fixture
def cond_other_fit_bytes():
    other = ConditionalOtherBN(
        ["C", "D"],
        ["A", "B"],
        [("A", "C")],
        [("C", pbn.CKDEType()), ("D", pbn.DiscreteFactorType())],
    )
    cpd_c = pbn.CKDE("C", ["A"])
    cpd_d = pbn.DiscreteFactor("D", [])

    df_continuous = generate_normal_data_independent(100)
    cpd_c.fit(df_continuous)

    df_discrete = generate_discrete_data(100)
    cpd_d = pbn.DiscreteFactor("D", [])
    cpd_d.fit(df_discrete)

    other.add_cpds([cpd_c, cpd_d])

    other.include_cpd = True
    return pickle.dumps(other)


def test_serialization_fitted_conditional_bn(
    cond_gaussian_partial_fit_bytes,
    cond_gaussian_fit_bytes,
    cond_other_partial_fit_bytes,
    cond_other_fit_bytes,
):
    # ####################
    # Gaussian partial fit
    # ####################
    loaded_partial = pickle.loads(cond_gaussian_partial_fit_bytes)
    assert not loaded_partial.fitted()
    cpd = loaded_partial.cpd("C")
    assert cpd.variable() == "C"
    assert cpd.evidence() == ["A"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    # ####################
    # Gaussian fit
    # ####################
    loaded_fitted = pickle.loads(cond_gaussian_fit_bytes)
    assert loaded_fitted.fitted()

    cpd_c = loaded_fitted.cpd("C")
    assert cpd_c.variable() == "C"
    assert cpd_c.evidence() == ["A"]
    assert list(cpd_c.beta) == [1, 2]
    assert cpd_c.variance == 2

    cpd_d = loaded_fitted.cpd("D")
    assert cpd_d.variable() == "D"
    assert cpd_d.evidence() == []
    assert cpd_d.beta == [3]
    assert cpd_d.variance == 1.5

    # ####################
    # OtherBN homogeneous partial fit
    # ####################
    loaded_other = pickle.loads(cond_other_partial_fit_bytes)
    assert not loaded_other.fitted()
    cpd = loaded_other.cpd("D")
    assert cpd.variable() == "D"
    assert cpd.evidence() == []
    assert cpd.beta == [3]
    assert cpd.variance == 1.5

    # ####################
    # OtherBN homogeneous fit
    # ####################
    loaded_other_fitted = pickle.loads(cond_other_fit_bytes)
    assert loaded_other_fitted.fitted()

    cpd_c = loaded_other_fitted.cpd("C")
    assert cpd_c.variable() == "C"
    assert cpd_c.evidence() == ["A"]
    assert cpd_c.fitted()
    assert cpd_c.num_instances() == 100
    assert cpd_c.type() == pbn.CKDEType()

    cpd_d = loaded_other_fitted.cpd("D")
    assert cpd_d.variable() == "D"
    assert cpd_d.evidence() == []
    assert cpd_d.fitted()
    assert cpd_d.type() == pbn.DiscreteFactorType()

    assert loaded_other_fitted.extra_info == "extra"
    assert loaded_other.type() == loaded_other_fitted.type()


# ##########################
# Dynamic BN
# ##########################


@pytest.fixture
def dyn_gaussian_bytes():
    gaussian = pbn.DynamicGaussianNetwork(["A", "B", "C", "D"], 2)
    gaussian.static_bn().add_arc("A_t_2", "D_t_1")
    gaussian.transition_bn().add_arc("C_t_2", "B_t_0")
    return pickle.dumps(gaussian)


@pytest.fixture
def dyn_spbn_bytes():
    spbn = pbn.DynamicSemiparametricBN(["A", "B", "C", "D"], 2)
    spbn.static_bn().add_arc("A_t_2", "D_t_1")
    spbn.transition_bn().add_arc("C_t_2", "B_t_0")
    spbn.transition_bn().set_node_type("B_t_0", pbn.CKDEType())
    return pickle.dumps(spbn)


@pytest.fixture
def dyn_kde_bytes():
    kde = pbn.DynamicKDENetwork(["A", "B", "C", "D"], 2)
    kde.static_bn().add_arc("A_t_2", "D_t_1")
    kde.transition_bn().add_arc("C_t_2", "B_t_0")
    return pickle.dumps(kde)


@pytest.fixture
def dyn_discrete_bytes():
    discrete = pbn.DynamicDiscreteBN(["A", "B", "C", "D"], 2)
    discrete.static_bn().add_arc("A_t_2", "D_t_1")
    discrete.transition_bn().add_arc("C_t_2", "B_t_0")
    return pickle.dumps(discrete)


@pytest.fixture
def dyn_genericbn_bytes():
    gen = pbn.DynamicBayesianNetwork(
        MyRestrictedGaussianNetworkType(), ["A", "B", "C", "D"], 2
    )
    gen.static_bn().add_arc("A_t_2", "D_t_1")
    gen.transition_bn().add_arc("A_t_2", "B_t_0")
    return pickle.dumps(gen)


class DynamicNewBN(pbn.DynamicBayesianNetwork):
    def __init__(self, variables, markovian_order):
        pbn.DynamicBayesianNetwork.__init__(
            self, MyRestrictedGaussianNetworkType(), variables, markovian_order
        )


class DynamicOtherBN(pbn.DynamicBayesianNetwork):
    def __init__(self, variables, markovian_order, static_bn=None, transition_bn=None):
        if static_bn is None or transition_bn is None:
            pbn.DynamicBayesianNetwork.__init__(
                self, NonHomogeneousType(), variables, markovian_order
            )
        else:
            pbn.DynamicBayesianNetwork.__init__(
                self, variables, markovian_order, static_bn, transition_bn
            )
        self.extra_info = "extra"

    def __getstate_extra__(self):
        return self.extra_info

    def __setstate_extra__(self, t):
        self.extra_info = t


@pytest.fixture
def dyn_newbn_bytes():
    new = DynamicNewBN(["A", "B", "C", "D"], 2)
    new.static_bn().add_arc("A_t_2", "D_t_1")
    new.transition_bn().add_arc("A_t_2", "B_t_0")
    return pickle.dumps(new)


@pytest.fixture
def dyn_otherbn_bytes():
    other = DynamicOtherBN(["A", "B", "C", "D"], 2)
    other.static_bn().add_arc("A_t_2", "D_t_1")
    other.static_bn().set_node_type("C_t_1", pbn.DiscreteFactorType())
    other.static_bn().set_node_type("D_t_1", pbn.CKDEType())

    other.transition_bn().add_arc("A_t_2", "B_t_0")
    other.transition_bn().set_node_type("D_t_0", pbn.CKDEType())
    return pickle.dumps(other)


def test_serialization_dbn_model(
    dyn_gaussian_bytes,
    dyn_spbn_bytes,
    dyn_kde_bytes,
    dyn_discrete_bytes,
    dyn_genericbn_bytes,
    dyn_newbn_bytes,
    dyn_otherbn_bytes,
):
    loaded_g = pickle.loads(dyn_gaussian_bytes)
    assert set(loaded_g.variables()) == set(["A", "B", "C", "D"])
    assert loaded_g.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_g.transition_bn().arcs() == [("C_t_2", "B_t_0")]
    assert loaded_g.type() == pbn.GaussianNetworkType()

    loaded_s = pickle.loads(dyn_spbn_bytes)
    assert set(loaded_s.variables()) == set(["A", "B", "C", "D"])
    assert loaded_s.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_s.transition_bn().arcs() == [("C_t_2", "B_t_0")]
    assert loaded_s.type() == pbn.SemiparametricBNType()
    node_types = {v + "_t_0": pbn.UnknownFactorType() for v in loaded_s.variables()}
    node_types["B_t_0"] = pbn.CKDEType()
    assert loaded_s.transition_bn().node_types() == node_types

    loaded_k = pickle.loads(dyn_kde_bytes)
    assert set(loaded_k.variables()) == set(["A", "B", "C", "D"])
    assert loaded_k.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_k.transition_bn().arcs() == [("C_t_2", "B_t_0")]
    assert loaded_k.type() == pbn.KDENetworkType()

    loaded_d = pickle.loads(dyn_discrete_bytes)
    assert set(loaded_d.variables()) == set(["A", "B", "C", "D"])
    assert loaded_d.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_d.transition_bn().arcs() == [("C_t_2", "B_t_0")]
    assert loaded_d.type() == pbn.DiscreteBNType()

    loaded_gen = pickle.loads(dyn_genericbn_bytes)
    assert set(loaded_gen.variables()) == set(["A", "B", "C", "D"])
    assert loaded_gen.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_gen.transition_bn().arcs() == [("A_t_2", "B_t_0")]
    assert loaded_gen.type() == MyRestrictedGaussianNetworkType()

    loaded_nn = pickle.loads(dyn_newbn_bytes)
    assert set(loaded_nn.variables()) == set(["A", "B", "C", "D"])
    assert loaded_nn.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_nn.transition_bn().arcs() == [("A_t_2", "B_t_0")]
    assert loaded_nn.type() == MyRestrictedGaussianNetworkType()

    loaded_other = pickle.loads(dyn_otherbn_bytes)
    assert set(loaded_other.variables()) == set(["A", "B", "C", "D"])
    assert loaded_other.static_bn().arcs() == [("A_t_2", "D_t_1")]
    assert loaded_other.transition_bn().arcs() == [("A_t_2", "B_t_0")]
    assert loaded_other.type() == NonHomogeneousType()
    assert loaded_other.extra_info == "extra"

    assert loaded_other.static_bn().node_type("C_t_1") == pbn.DiscreteFactorType()
    assert loaded_other.static_bn().node_type("D_t_1") == pbn.CKDEType()
    assert loaded_other.transition_bn().node_type("D_t_0") == pbn.CKDEType()


@pytest.fixture
def dyn_gaussian_partial_fit_bytes():
    gaussian = pbn.DynamicGaussianNetwork(["A", "B", "C", "D"], 2)
    gaussian.static_bn().add_arc("A_t_2", "D_t_1")
    gaussian.transition_bn().add_arc("C_t_2", "B_t_0")
    lg = pbn.LinearGaussianCPD("D_t_1", ["A_t_2"], [1, 2], 2)
    gaussian.static_bn().add_cpds([lg])
    lg = pbn.LinearGaussianCPD("B_t_0", ["C_t_2"], [3, 4], 5)
    gaussian.transition_bn().add_cpds([lg])
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)


@pytest.fixture
def dyn_gaussian_fit_bytes():
    gaussian = pbn.DynamicGaussianNetwork(["A", "B", "C", "D"], 2)
    gaussian.static_bn().add_arc("A_t_2", "D_t_1")
    gaussian.transition_bn().add_arc("C_t_2", "B_t_0")
    df = generate_normal_data_independent(1000)
    gaussian.fit(df)
    gaussian.include_cpd = True
    return pickle.dumps(gaussian)


@pytest.fixture
def dyn_other_partial_fit_bytes():
    variables = ["A", "B", "C", "D"]
    static_nodes = [v + "_t_" + str(m) for v in variables for m in range(1, 3)]
    transition_nodes = [v + "_t_0" for v in variables]

    other_static = OtherBN(
        static_nodes,
        [("A_t_2", "D_t_1")],
        [
            ("B_t_1", pbn.DiscreteFactorType()),
            ("C_t_1", pbn.CKDEType()),
            ("D_t_1", pbn.LinearGaussianCPDType()),
        ],
    )
    lg = pbn.LinearGaussianCPD("D_t_1", ["A_t_2"], [1, 2], 2)
    other_static.add_cpds([lg])

    other_transition = ConditionalOtherBN(
        transition_nodes,
        static_nodes,
        [("A_t_2", "D_t_0")],
        [
            ("B_t_0", pbn.DiscreteFactorType()),
            ("C_t_0", pbn.CKDEType()),
            ("D_t_0", pbn.LinearGaussianCPDType()),
        ],
    )
    lg = pbn.LinearGaussianCPD("D_t_0", ["A_t_2"], [3, 4], 1.5)
    other_transition.add_cpds([lg])

    assert other_static.type() == other_transition.type()

    dyn_other = DynamicOtherBN(variables, 2, other_static, other_transition)
    dyn_other.include_cpd = True
    return pickle.dumps(dyn_other)


@pytest.fixture
def dyn_other_fit_bytes():
    variables = ["A", "B", "C", "D"]
    static_nodes = [v + "_t_" + str(m) for v in variables for m in range(1, 3)]
    transition_nodes = [v + "_t_0" for v in variables]

    other_static = OtherBN(
        static_nodes,
        [("A_t_2", "D_t_1")],
        [
            ("B_t_2", pbn.DiscreteFactorType()),
            ("B_t_1", pbn.DiscreteFactorType()),
            ("C_t_1", pbn.CKDEType()),
            ("D_t_1", pbn.LinearGaussianCPDType()),
        ],
    )
    lg = pbn.LinearGaussianCPD("D_t_1", ["A_t_2"], [1, 2], 2)
    other_static.add_cpds([lg])

    other_transition = ConditionalOtherBN(
        transition_nodes,
        static_nodes,
        [("A_t_2", "D_t_0")],
        [
            ("B_t_0", pbn.DiscreteFactorType()),
            ("C_t_0", pbn.CKDEType()),
            ("D_t_0", pbn.LinearGaussianCPDType()),
        ],
    )
    lg = pbn.LinearGaussianCPD("D_t_0", ["A_t_2"], [3, 4], 1.5)
    other_transition.add_cpds([lg])

    assert other_static.type() == other_transition.type()

    dyn_other = DynamicOtherBN(variables, 2, other_static, other_transition)
    df_continuous = generate_normal_data_independent(1000)
    df_discrete = generate_discrete_data(1000)
    df = df_continuous
    df["B"] = df_discrete["B"]
    dyn_other.fit(df)
    dyn_other.include_cpd = True
    return pickle.dumps(dyn_other)


def test_serialization_fitted_dbn(
    dyn_gaussian_partial_fit_bytes,
    dyn_gaussian_fit_bytes,
    dyn_other_partial_fit_bytes,
    dyn_other_fit_bytes,
):
    # ####################
    # Gaussian partial fit
    # ####################
    loaded_partial = pickle.loads(dyn_gaussian_partial_fit_bytes)
    assert not loaded_partial.fitted()
    assert not loaded_partial.static_bn().fitted()
    assert not loaded_partial.transition_bn().fitted()
    cpd = loaded_partial.static_bn().cpd("D_t_1")
    assert cpd.variable() == "D_t_1"
    assert cpd.evidence() == ["A_t_2"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    cpd = loaded_partial.transition_bn().cpd("B_t_0")
    assert cpd.variable() == "B_t_0"
    assert cpd.evidence() == ["C_t_2"]
    assert list(cpd.beta) == [3, 4]
    assert cpd.variance == 5

    # ####################
    # Gaussian fit
    # ####################
    loaded_fitted = pickle.loads(dyn_gaussian_fit_bytes)
    assert loaded_fitted.fitted()
    assert loaded_fitted.static_bn().fitted()
    assert loaded_fitted.transition_bn().fitted()

    # ####################
    # Other partial fit
    # ####################
    loaded_partial = pickle.loads(dyn_other_partial_fit_bytes)
    assert not loaded_partial.fitted()
    assert not loaded_partial.static_bn().fitted()
    assert not loaded_partial.transition_bn().fitted()
    assert loaded_partial.static_bn().node_type("B_t_1") == pbn.DiscreteFactorType()
    assert loaded_partial.static_bn().node_type("C_t_1") == pbn.CKDEType()
    assert loaded_partial.static_bn().node_type("D_t_1") == pbn.LinearGaussianCPDType()

    assert loaded_partial.transition_bn().node_type("B_t_0") == pbn.DiscreteFactorType()
    assert loaded_partial.transition_bn().node_type("C_t_0") == pbn.CKDEType()
    assert (
        loaded_partial.transition_bn().node_type("D_t_0") == pbn.LinearGaussianCPDType()
    )

    cpd = loaded_partial.static_bn().cpd("D_t_1")
    assert cpd.variable() == "D_t_1"
    assert cpd.evidence() == ["A_t_2"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    cpd = loaded_partial.transition_bn().cpd("D_t_0")
    assert cpd.variable() == "D_t_0"
    assert cpd.evidence() == ["A_t_2"]
    assert list(cpd.beta) == [3, 4]
    assert cpd.variance == 1.5

    # ####################
    # Other fit
    # ####################
    loaded_fitted = pickle.loads(dyn_other_fit_bytes)
    assert loaded_fitted.fitted()
    assert loaded_fitted.static_bn().fitted()
    assert loaded_fitted.transition_bn().fitted()
    assert loaded_partial.static_bn().node_type("B_t_1") == pbn.DiscreteFactorType()
    assert loaded_partial.static_bn().node_type("C_t_1") == pbn.CKDEType()
    assert loaded_partial.static_bn().node_type("D_t_1") == pbn.LinearGaussianCPDType()

    assert loaded_partial.transition_bn().node_type("B_t_0") == pbn.DiscreteFactorType()
    assert loaded_partial.transition_bn().node_type("C_t_0") == pbn.CKDEType()
    assert (
        loaded_partial.transition_bn().node_type("D_t_0") == pbn.LinearGaussianCPDType()
    )

    cpd = loaded_partial.static_bn().cpd("D_t_1")
    assert cpd.variable() == "D_t_1"
    assert cpd.evidence() == ["A_t_2"]
    assert list(cpd.beta) == [1, 2]
    assert cpd.variance == 2

    cpd = loaded_partial.transition_bn().cpd("D_t_0")
    assert cpd.variable() == "D_t_0"
    assert cpd.evidence() == ["A_t_2"]
    assert list(cpd.beta) == [3, 4]
    assert cpd.variance == 1.5
