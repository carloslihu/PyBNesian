#ifndef PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP
#define PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP

#include <indicators/cursor_control.hpp>
#include <dataset/dataset.hpp>
#include <learning/scores/scores.hpp>
#include <learning/operators/operators.hpp>
#include <util/progress.hpp>
#include <util/vector.hpp>

namespace py = pybind11; 

using dataset::DataFrame;
using learning::scores::Score;
using learning::operators::Operator, learning::operators::OperatorType, learning::operators::ArcOperator, 
      learning::operators::ChangeNodeType, learning::operators::OperatorTabuSet, learning::operators::OperatorPool;

using util::ArcStringVector;

namespace learning::algorithms {

    // TODO: Include start graph.
    std::unique_ptr<BayesianNetworkBase> hc(const DataFrame& df, 
                                            const BayesianNetworkBase* start,
                                            std::string bn_str, 
                                            std::string score_str,
                                            std::vector<std::string> operators_str,
                                            ArcStringVector& arc_blacklist,
                                            ArcStringVector& arc_whitelist,
                                            FactorStringTypeVector& type_whitelist,
                                            int max_indegree,
                                            int max_iters,
                                            double epsilon,
                                            int patience,
                                            std::optional<unsigned int> seed,
                                            int num_folds,
                                            double test_holdout_ratio,
                                            int verbose = 0);

    class GreedyHillClimbing {
    public:
        std::unique_ptr<BayesianNetworkBase> estimate(OperatorPool& op_pool,
                                                      const BayesianNetworkBase& start,
                                                      ArcStringVector& arc_blacklist,
                                                      ArcStringVector& arc_whitelist,
                                                      int max_indegree,
                                                      int max_iters, 
                                                      double epsilon,
                                                      int verbose = 0);

        std::unique_ptr<BayesianNetworkBase> estimate_validation(OperatorPool& op_pool,
                                                                 Score& validation_score,
                                                                 const BayesianNetworkBase& start,
                                                                 ArcStringVector& arc_blacklist,
                                                                 ArcStringVector& arc_whitelist,
                                                                 FactorStringTypeVector& type_whitelist,
                                                                 int max_indegree,
                                                                 int max_iters,
                                                                 double epsilon, 
                                                                 int patience,
                                                                 int verbose = 0);
    };
}

#endif //PYBNESIAN_LEARNING_ALGORITHMS_HILLCLIMBING_HPP