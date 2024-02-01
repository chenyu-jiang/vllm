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
      const GraphNodes& ready_nodes, float current_time) override;

  float GetAvgActivatedExperts() const override;
  float GetAvgBatchSize() const override;

 private:
  void AdvancePhase_();
  void ResetBatch_();

  void UpdateAvgActivatedExperts_(int n_experts);
  void UpdateAvgBatchSize_(int batch_size);

  int n_layers_ = 0;
  int max_batch_size_ = 0;
  int max_kv_tokens_ = 0;
  float per_token_latency_slo_ = 0.0;
  std::set<std::pair<int, int>> current_batch_requests_;
  std::set<std::pair<int, int>> prev_batch_requests_;
  int current_layer_ = 0;
  NodeType current_phase_ = NodeType::kAttn;
  int activated_experts_per_layer_ = 0;
  std::pair<float, int> avg_activated_experts_;
  std::pair<float, int> avg_batch_size_;
};

REGISTER_STRATEGY(FCFS, FCFSStrategy);

}  // namespace fcfs_strategy
}  // namespace stragegies