#pragma once

#include <set>

#include "strategy_factory.hpp"

namespace stragegies {
namespace lwg_strategy {

using dependency_graph::NodeType;

class LayerWiseGreedyStrategy : public Strategy {
 public:
  LayerWiseGreedyStrategy(const RequestGraphs& graphs,
                          const StrategyConfig& config);
  ScheduleResult Schedule(
      const std::unordered_map<int, RequestStats>& request_stats,
      const GraphNodes& ready_nodes, float current_time) override;

  float GetAvgActivatedExperts() const override;
  float GetAvgBatchSize() const override;

 private:
  void AdvancePhase_();
  void UpdateAvgActivatedExperts_(int n_experts);
  void UpdateAvgBatchSize_(int batch_size);

  int n_layers_ = 0;
  int max_batch_size_ = 0;
  float per_token_latency_slo_ = 0.0;
  int k_experts_per_token_ = 0;
  int current_layer_id_ = 0;
  std::set<int> current_req_ids_;
  NodeType current_phase_ = NodeType::kAttn;
  std::pair<float, int> avg_activated_experts_;
  std::pair<float, int> avg_batch_size_;
};

REGISTER_STRATEGY(LWG, LayerWiseGreedyStrategy);

}  // namespace lwg_strategy
}  // namespace stragegies