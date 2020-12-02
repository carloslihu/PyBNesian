#ifndef PYBNESIAN_LEARNING_SCORES_SCORES_HPP
#define PYBNESIAN_LEARNING_SCORES_SCORES_HPP

#include <models/GaussianNetwork.hpp>
#include <models/SemiparametricBN.hpp>

using models::BayesianNetworkBase, models::GaussianNetwork, models::SemiparametricBN;

namespace learning::scores {

    class ScoreType
    {
    public:
        enum Value : uint8_t
        {
            BIC,
            PREDICTIVE_LIKELIHOOD,
            HOLDOUT_LIKELIHOOD
        };

        struct Hash
        {
            inline std::size_t operator ()(ScoreType const score_type) const
            {
                return static_cast<std::size_t>(score_type.value);
            }
        };

        using HashType = Hash;

        ScoreType() = default;
        constexpr ScoreType(Value opset_type) : value(opset_type) { }

        operator Value() const { return value; }  
        explicit operator bool() = delete;

        constexpr bool operator==(ScoreType a) const { return value == a.value; }
        constexpr bool operator==(Value v) const { return value == v; }
        constexpr bool operator!=(ScoreType a) const { return value != a.value; }
        constexpr bool operator!=(Value v) const { return value != v; }

        std::string ToString() const { 
            switch(value) {
                case Value::BIC:
                    return "bic";
                case Value::PREDICTIVE_LIKELIHOOD:
                    return "predic-l";
                case Value::HOLDOUT_LIKELIHOOD:
                    return "holdout-l";
                default:
                    throw std::invalid_argument("Unreachable code in ScoreType.");
            }
        }

    private:
        Value value;
    };

    class Score {
    public:
        virtual ~Score() {}
        virtual double score(const BayesianNetworkBase& model) const {
            double s = 0;
            for (auto node = 0; node < model.num_nodes(); ++node) {
                s += local_score(model, node);
            }

            return s;
        }

        virtual double local_score(const BayesianNetworkBase&, int) const = 0;
        virtual double local_score(const BayesianNetworkBase&, const std::string&) const = 0;
        virtual double local_score(const BayesianNetworkBase&, int,
                                    const typename std::vector<int>::const_iterator, 
                                    const typename std::vector<int>::const_iterator) const = 0;
        virtual double local_score(const BayesianNetworkBase&, const std::string&,
                                    const typename std::vector<std::string>::const_iterator, 
                                    const typename std::vector<std::string>::const_iterator) const = 0;

        virtual std::string ToString() const = 0;
        virtual bool is_decomposable() const = 0;
        virtual ScoreType type() const = 0;
    };

    class ScoreSPBN {
    public:
        virtual ~ScoreSPBN() {}
        virtual double local_score(FactorType variable_type, int variable, 
                                   const typename std::vector<int>::const_iterator evidence_begin, 
                                   const typename std::vector<int>::const_iterator evidence_end) const = 0;
        virtual double local_score(FactorType variable_type, const std::string& variable, 
                                   const typename std::vector<std::string>::const_iterator evidence_begin, 
                                   const typename std::vector<std::string>::const_iterator evidence_end) const = 0;
    };
}


#endif //PYBNESIAN_LEARNING_SCORES_SCORES_HPP