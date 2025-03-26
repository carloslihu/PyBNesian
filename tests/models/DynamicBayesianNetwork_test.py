import re

import numpy as np
import pandas as pd
import pybnesian as pbn
import pytest
from helpers.data import generate_normal_data
from scipy.stats import norm

df = generate_normal_data(1000)


def test_create_dbn():
    variables = ["A", "B", "C", "D"]
    gbn = pbn.DynamicGaussianNetwork(variables, 2)

    assert gbn.markovian_order() == 2
    assert gbn.variables() == ["A", "B", "C", "D"]
    assert gbn.num_variables() == 4
    assert gbn.type() == pbn.GaussianNetworkType()

    transition_nodes = [v + "_t_0" for v in variables]
    static_nodes = [v + "_t_" + str(m) for v in variables for m in range(1, 3)]

    assert set(gbn.static_bn().nodes()) == set(static_nodes)
    assert set(gbn.transition_bn().interface_nodes()) == set(static_nodes)
    assert set(gbn.transition_bn().nodes()) == set(transition_nodes)

    static_bn = pbn.GaussianNetwork(static_nodes)
    transition_bn = pbn.ConditionalGaussianNetwork(transition_nodes, static_nodes)

    gbn2 = pbn.DynamicGaussianNetwork(variables, 2, static_bn, transition_bn)
    assert gbn2.markovian_order() == 2
    assert gbn2.variables() == ["A", "B", "C", "D"]
    assert gbn2.num_variables() == 4
    assert gbn2.type() == pbn.GaussianNetworkType()

    wrong_transition_bn = pbn.ConditionalDiscreteBN(transition_nodes, static_nodes)

    with pytest.raises(ValueError) as ex:
        pbn.DynamicGaussianNetwork(variables, 2, static_bn, wrong_transition_bn)
    assert "Static and transition Bayesian networks do not have the same type" in str(
        ex.value
    )

    wrong_static_bn = pbn.DiscreteBN(static_nodes)
    with pytest.raises(ValueError) as ex:
        pbn.DynamicGaussianNetwork(variables, 2, wrong_static_bn, wrong_transition_bn)
    assert "Bayesian networks are not Gaussian." in str(ex.value)


def test_variable_operations_dbn():
    variables = ["A", "B", "C", "D"]
    gbn = pbn.DynamicGaussianNetwork(variables, 2)

    assert gbn.markovian_order() == 2
    assert gbn.variables() == ["A", "B", "C", "D"]
    assert gbn.num_variables() == 4

    assert gbn.contains_variable("A")
    assert gbn.contains_variable("B")
    assert gbn.contains_variable("C")
    assert gbn.contains_variable("D")

    gbn.add_variable("E")
    assert set(gbn.variables()) == set(["A", "B", "C", "D", "E"])
    assert gbn.num_variables() == 5

    assert set(gbn.static_bn().nodes()) == set(
        [v + "_t_" + str(m) for v in variables + ["E"] for m in range(1, 3)]
    )
    assert set(gbn.transition_bn().nodes()) == set(
        [v + "_t_0" for v in variables + ["E"]]
    )

    gbn.remove_variable("B")
    assert set(gbn.variables()) == set(["A", "C", "D", "E"])
    assert gbn.num_variables() == 4
    assert set(gbn.static_bn().nodes()) == set(
        [v + "_t_" + str(m) for v in ["A", "C", "D", "E"] for m in range(1, 3)]
    )
    assert set(gbn.transition_bn().nodes()) == set(
        [v + "_t_0" for v in ["A", "C", "D", "E"]]
    )


def test_fit_dbn():
    variables = ["A", "B", "C", "D"]
    gbn = pbn.DynamicGaussianNetwork(variables, 2)
    assert not gbn.fitted()
    assert not gbn.static_bn().fitted()
    assert not gbn.transition_bn().fitted()
    gbn.fit(df)
    assert gbn.fitted()

    ddf = pbn.DynamicDataFrame(df, 2)
    gbn2 = pbn.DynamicGaussianNetwork(variables, 2)
    gbn2.static_bn().fit(ddf.static_df())
    assert not gbn2.fitted()
    assert gbn2.static_bn().fitted()
    assert not gbn2.transition_bn().fitted()

    gbn2.transition_bn().fit(ddf.transition_df())
    assert gbn2.fitted()
    assert gbn2.static_bn().fitted()
    assert gbn2.transition_bn().fitted()


def lg_logl_row(row, variable, evidence, beta, variance):
    m = beta[0] + beta[1:].dot(row[evidence])
    return norm(m, np.sqrt(variance)).logpdf(row[variable])


def static_logl(dbn, test_data, index, variable):
    sl = test_data.head(dbn.markovian_order())

    node_name = variable + "_t_" + str(dbn.markovian_order() - index)
    cpd = dbn.static_bn().cpd(node_name)
    evidence = cpd.evidence()

    row_values = [sl.loc[index, variable]]
    for e in evidence:
        m = re.search("(.*)_t_(\\d+)", e)
        if m:
            e_var = m.group(1)
            t = int(m.group(2))

            row_values.append(sl.loc[dbn.markovian_order() - t, e_var])

    r = pd.Series(data=row_values, index=[node_name] + evidence)

    return lg_logl_row(r, node_name, evidence, cpd.beta, cpd.variance)


def transition_logl(dbn, test_data, index, variable):
    node_name = variable + "_t_0"
    cpd = dbn.transition_bn().cpd(node_name)
    evidence = cpd.evidence()

    row_values = [test_data.loc[index, variable]]
    for e in evidence:
        m = re.search("(.*)_t_(\\d+)", e)
        if m:
            e_var = m.group(1)
            t = int(m.group(2))

            row_values.append(test_data.loc[index - t, e_var])

    r = pd.Series(data=row_values, index=[node_name] + evidence)
    return lg_logl_row(r, node_name, evidence, cpd.beta, cpd.variance)


def numpy_logl(dbn, test_data):
    ll = np.zeros((test_data.shape[0],))

    for i in range(dbn.markovian_order()):
        for v in dbn.variables():
            ll[i] += static_logl(dbn, test_data, i, v)

    for i in range(dbn.markovian_order(), test_data.shape[0]):
        for v in dbn.variables():
            ll[i] += transition_logl(dbn, test_data, i, v)

    return ll


def test_logl_dbn():
    variables = ["A", "B", "C", "D"]

    static_bn = pbn.GaussianNetwork(
        ["A", "B", "C", "D"], [("A", "C"), ("B", "C"), ("C", "D")]
    )
    static_bn = pbn.GaussianNetwork(
        ["A", "B", "C", "D"], [("A", "C"), ("B", "C"), ("C", "D")]
    )
    gbn = pbn.DynamicGaussianNetwork(variables, 2)

    static_bn = gbn.static_bn()
    static_bn.add_arc("A_t_2", "C_t_2")
    static_bn.add_arc("B_t_2", "C_t_2")
    static_bn.add_arc("C_t_2", "D_t_2")
    static_bn.add_arc("A_t_1", "C_t_1")
    static_bn.add_arc("B_t_1", "C_t_1")
    static_bn.add_arc("C_t_1", "D_t_1")

    transition_bn = gbn.transition_bn()
    transition_bn.add_arc("A_t_2", "A_t_0")
    transition_bn.add_arc("B_t_2", "B_t_0")
    transition_bn.add_arc("C_t_2", "C_t_0")
    transition_bn.add_arc("D_t_2", "D_t_0")
    transition_bn.add_arc("A_t_1", "A_t_0")
    transition_bn.add_arc("B_t_1", "B_t_0")
    transition_bn.add_arc("C_t_1", "C_t_0")
    transition_bn.add_arc("D_t_1", "D_t_0")

    gbn.fit(df)

    test_df = generate_normal_data(100)
    ground_truth_ll = numpy_logl(gbn, generate_normal_data(100))
    ll = gbn.logl(test_df)
    assert np.all(np.isclose(ground_truth_ll, ll))


def test_slogl_dbn():
    variables = ["A", "B", "C", "D"]

    static_bn = pbn.GaussianNetwork(
        ["A", "B", "C", "D"], [("A", "C"), ("B", "C"), ("C", "D")]
    )
    static_bn = pbn.GaussianNetwork(
        ["A", "B", "C", "D"], [("A", "C"), ("B", "C"), ("C", "D")]
    )
    gbn = pbn.DynamicGaussianNetwork(variables, 2)

    static_bn = gbn.static_bn()
    static_bn.add_arc("A_t_2", "C_t_2")
    static_bn.add_arc("B_t_2", "C_t_2")
    static_bn.add_arc("C_t_2", "D_t_2")
    static_bn.add_arc("A_t_1", "C_t_1")
    static_bn.add_arc("B_t_1", "C_t_1")
    static_bn.add_arc("C_t_1", "D_t_1")

    transition_bn = gbn.transition_bn()
    transition_bn.add_arc("A_t_2", "A_t_0")
    transition_bn.add_arc("B_t_2", "B_t_0")
    transition_bn.add_arc("C_t_2", "C_t_0")
    transition_bn.add_arc("D_t_2", "D_t_0")
    transition_bn.add_arc("A_t_1", "A_t_0")
    transition_bn.add_arc("B_t_1", "B_t_0")
    transition_bn.add_arc("C_t_1", "C_t_0")
    transition_bn.add_arc("D_t_1", "D_t_0")

    gbn.fit(df)
    test_df = generate_normal_data(100)
    ll = numpy_logl(gbn, test_df)
    assert np.isclose(gbn.slogl(test_df), ll.sum())
