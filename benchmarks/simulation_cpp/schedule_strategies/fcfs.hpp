#pragma once

#include <set>

#include "strategy_factory.hpp"

namespace stragegies {
namespace fcfs_strategy {

using dependency_graph::NodeType;

class FCFSStrategy : public Strategy {
 public:
  FCFSStrategy(const RequestGraphs& graphs, const StrategyConfig& config);
  ScheduleResult Schedule(
      const std::unordered_map<int, RequestStats>& request_stats,
      const GraphNodes& ready_nodes, int max_batch_size = -1) override;

  float GetAvgActivatedExperts() const override;

 private:
  void AdvancePhase_();
  void ResetBatch_();

  int n_layers_ = 0;
  float per_token_latency_slo_ = 0.0;
  std::set<std::pair<int, int>> current_batch_requests_;
  std::set<std::pair<int, int>> prev_batch_requests_;
  int current_layer_ = 0;
  NodeType current_phase_ = NodeType::kAttn;
  int activated_experts_per_batch_ = 0;
  std::vector<int> activated_experts_history_;
};

REGISTER_STRATEGY(FCFS, FCFSStrategy);

}  // namespace fcfs_strategy
}  // namespace stragegies