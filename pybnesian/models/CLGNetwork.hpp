#ifndef PYBNESIAN_MODELS_CLGNETWORK_HPP
#define PYBNESIAN_MODELS_CLGNETWORK_HPP

#include <factors/continuous/LinearGaussianCPD.hpp>
#include <factors/discrete/DiscreteFactor.hpp>
#include <models/BayesianNetwork.hpp>
#include <models/DynamicBayesianNetwork.hpp>

using factors::continuous::LinearGaussianCPDType;
using factors::discrete::DiscreteFactorType;

namespace models {

class CLGNetworkType : public BayesianNetworkType {
public:
    CLGNetworkType(const CLGNetworkType&) = delete;
    void operator=(const CLGNetworkType&) = delete;

    static std::shared_ptr<CLGNetworkType> get() {
        static std::shared_ptr<CLGNetworkType> singleton = std::shared_ptr<CLGNetworkType>(new CLGNetworkType);
        return singleton;
    }

    static CLGNetworkType& get_ref() {
        static CLGNetworkType& ref = *CLGNetworkType::get();
        return ref;
    }

    std::shared_ptr<BayesianNetworkBase> new_bn(const std::vector<std::string>& nodes) const override;
    std::shared_ptr<ConditionalBayesianNetworkBase> new_cbn(
        const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const override;

    bool is_homogeneous() const override { return false; }

    std::shared_ptr<FactorType> default_node_type() const override { return LinearGaussianCPDType::get(); }

    std::shared_ptr<FactorType> data_default_node_type(const std::shared_ptr<DataType>& dt) const override {
        switch (dt->id()) {
            case Type::DOUBLE:
            case Type::FLOAT:
                return LinearGaussianCPDType::get();
            case Type::DICTIONARY:
                return DiscreteFactorType::get();
            default:
                throw std::invalid_argument("Data type [" + dt->ToString() + "] not compatible with CLGNetworkType");
        }
    }

    bool compatible_node_type(const BayesianNetworkBase&,
                              const std::string&,
                              const std::shared_ptr<FactorType>& nt) const override {
        if (*nt != LinearGaussianCPDType::get_ref() && *nt != DiscreteFactorType::get_ref()) return false;

        return true;
    }

    bool compatible_node_type(const ConditionalBayesianNetworkBase& m,
                              const std::string& variable,
                              const std::shared_ptr<FactorType>& nt) const override {
        return compatible_node_type(static_cast<const BayesianNetworkBase&>(m), variable, nt);
    }

    bool can_have_arc(const BayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        return *m.node_type(target) == LinearGaussianCPDType::get_ref() ||
               *m.node_type(source) != LinearGaussianCPDType::get_ref();
    }

    bool can_have_arc(const ConditionalBayesianNetworkBase& m,
                      const std::string& source,
                      const std::string& target) const override {
        return can_have_arc(static_cast<const BayesianNetworkBase&>(m), source, target);
    }

    std::string ToString() const override { return "CLGNetworkType"; }

    py::tuple __getstate__() const override { return py::make_tuple(); }

    static std::shared_ptr<CLGNetworkType> __setstate__(py::tuple&) { return CLGNetworkType::get(); }

    static std::shared_ptr<CLGNetworkType> __setstate__(py::tuple&&) { return CLGNetworkType::get(); }

private:
    CLGNetworkType() { m_hash = reinterpret_cast<std::uintptr_t>(this); }
};

class CLGNetwork : public clone_inherit<CLGNetwork, BayesianNetwork> {
public:
    CLGNetwork(const std::vector<std::string>& nodes) : clone_inherit(CLGNetworkType::get(), nodes) {}

    CLGNetwork(const ArcStringVector& arcs) : clone_inherit(CLGNetworkType::get(), arcs) {}

    CLGNetwork(const std::vector<std::string>& nodes, const ArcStringVector& arcs)
        : clone_inherit(CLGNetworkType::get(), nodes, arcs) {}

    CLGNetwork(const Dag& graph) : clone_inherit(CLGNetworkType::get(), graph) {}
    CLGNetwork(Dag&& graph) : clone_inherit(CLGNetworkType::get(), std::move(graph)) {}

    std::string ToString() const override { return "CLGNetwork"; }
};

class ConditionalCLGNetwork : public clone_inherit<ConditionalCLGNetwork, ConditionalBayesianNetwork> {
public:
    ConditionalCLGNetwork(const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes)
        : clone_inherit(CLGNetworkType::get(), nodes, interface_nodes) {}

    ConditionalCLGNetwork(const std::vector<std::string>& nodes,
                          const std::vector<std::string>& interface_nodes,
                          const ArcStringVector& arcs)
        : clone_inherit(CLGNetworkType::get(), nodes, interface_nodes, arcs) {}

    ConditionalCLGNetwork(const ConditionalDag& graph) : clone_inherit(CLGNetworkType::get(), graph) {}
    ConditionalCLGNetwork(ConditionalDag&& graph) : clone_inherit(CLGNetworkType::get(), std::move(graph)) {}

    std::string ToString() const override { return "ConditionalCLGNetwork"; }
};

class DynamicCLGNetwork : public clone_inherit<DynamicCLGNetwork, DynamicBayesianNetwork> {
public:
    DynamicCLGNetwork(const std::vector<std::string>& variables, int markovian_order)
        : clone_inherit(CLGNetworkType::get(), variables, markovian_order) {}

    DynamicCLGNetwork(const std::vector<std::string>& variables,
                      int markovian_order,
                      std::shared_ptr<BayesianNetworkBase> static_bn,
                      std::shared_ptr<ConditionalBayesianNetworkBase> transition_bn)
        : clone_inherit(variables, markovian_order, static_bn, transition_bn) {
        if (static_bn->type_ref() != CLGNetworkType::get_ref())
            throw std::invalid_argument("Bayesian networks are not Gaussian.");
    }

    std::string ToString() const override { return "DynamicCLGNetwork"; }
};

}  // namespace models

#endif  // PYBNESIAN_MODELS_CLGNETWORK_HPP