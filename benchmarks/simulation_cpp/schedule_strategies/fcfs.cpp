#include "fcfs.hpp"

#include <algorithm>

namespace stragegies {
namespace fcfs_strategy {

using dependency_graph::ExpertNode;
using dependency_graph::GraphNode;

FCFSStrategy::FCFSStrategy(const RequestGraphs& graphs,
                           const StrategyConfig& config)
    : Strategy(graphs, config) {
  n_layers_ = config.GetInt("n_layers");
  per_token_latency_slo_ = config_.GetFloat("per_token_latency_slo");
}

ScheduleResult FCFSStrategy::Schedule(
    const std::unordered_map<int, RequestStats>& request_stats,
    const GraphNodes& ready_nodes, int max_batch_size) {
  if (ready_nodes.empty()) {
    // no more requests to schedule
    return ScheduleResult(ModelComponent::kAttnGate, {});
  }
  // execute the same requests layer by layer
  if (current_batch_requests_.empty()) {
    if (current_layer_ != 0) {
      throw std::runtime_error(
          "current_batch_requests_ is empty but current_layer_ is not 0");
    }
    // first try to execute the next token of the current batch
    GraphNodes ready_first_layer_attn_nodes;
    for (const auto& node : ready_nodes) {
      for (const auto& it : prev_batch_requests_) {
        if (node->req_id == it.first && node->token_index - 1 == it.second) {
          ready_first_layer_attn_nodes.push_back(node);
          current_batch_requests_.push_back({node->req_id, node->token_index});
        }
      }
    }
    if ((int)ready_first_layer_attn_nodes.size() < max_batch_size) {
      // schedule all first layer Attn requests in FCFS order
      // first sort ready nodes by enqueue time
      GraphNodes sorted_ready_nodes = ready_nodes;
      std::sort(sorted_ready_nodes.begin(), sorted_ready_nodes.end(),
                [&](const GraphNode* a, const GraphNode* b) {
                  return request_stats.at(a->req_id).enqueue_time < b->req_id;
                });
      for (const auto& node : sorted_ready_nodes) {
        if (node->layer_id == current_layer_ &&
            node->Type() == NodeType::kAttn) {
          for (const auto& it : current_batch_requests_) {
            if (node->req_id == it.first && node->token_index == it.second) {
              ready_first_layer_attn_nodes.push_back(node);
              current_batch_requests_.push_back(
                  {node->req_id, node->token_index});
              if ((int)ready_first_layer_attn_nodes.size() == max_batch_size) {
                break;
              }
            }
          }
        }
      }
    }
  }
  // first test if the current batch is all finished
  // we assume all nodes are DecodeNodes
  GraphNodes current_batch_ready_nodes;
  for (const auto& node : ready_nodes) {
    for (const auto& it : current_batch_requests_) {
      if (node->req_id == it.first && node->token_index == it.second) {
        current_batch_ready_nodes.push_back(node);
      }
    }
  }
  if (current_batch_ready_nodes.empty()) {
    // current batch is all finished
    // advance to the next phase
    ResetBatch_();
    Schedule(request_stats, ready_nodes, max_batch_size);
  }
  // filter out nodes that are not in the current phase
  GraphNodes current_phase_ready_nodes;
  for (const auto& node : current_batch_ready_nodes) {
    if (node->Type() == current_phase_ && node->layer_id == current_layer_) {
      current_phase_ready_nodes.push_back(node);
    }
  }
  if (current_phase_ready_nodes.empty()) {
    // current phase is all finished
    // advance to the next phase
    AdvancePhase_();
    current_phase_ready_nodes.clear();
    for (const auto& node : current_batch_ready_nodes) {
      if (node->Type() == current_phase_ && node->layer_id == current_layer_) {
        current_phase_ready_nodes.push_back(node);
      }
    }
  }
  // schedule the current phase
  if (current_phase_ == NodeType::kExpert) {
    // schedule one expert at a time
    std::vector<ExpertNode*> ready_expert_nodes;
    for (const auto& node : current_phase_ready_nodes) {
      ready_expert_nodes.push_back(dynamic_cast<ExpertNode*>(node));
    }
    int expert_id_to_schedule = ready_expert_nodes[0]->expert_id;
    GraphNodes nodes_to_schedule;
    for (const auto& node : ready_expert_nodes) {
      if (node->expert_id == expert_id_to_schedule) {
        nodes_to_schedule.push_back(node);
      }
    }
    activated_experts_per_batch_ += 1;
    return {ModelComponent::kExpert, nodes_to_schedule};
  }
  // schedule all attn nodes together
  return {ModelComponent::kAttnGate, current_phase_ready_nodes};
}

void FCFSStrategy::AdvancePhase_() {
  if (current_phase_ == NodeType::kAttn) {
    current_phase_ = NodeType::kExpert;
  } else {
    current_phase_ = NodeType::kAttn;
    current_layer_ += 1;
  }
}

void FCFSStrategy::ResetBatch_() {
  prev_batch_requests_ = current_batch_requests_;
  current_batch_requests_.clear();
  current_phase_ = NodeType::kAttn;
  current_layer_ = 0;
  activated_experts_history_.push_back(activated_experts_per_batch_);
  activated_experts_per_batch_ = 0;
}

float FCFSStrategy::GetAvgActivatedExperts() const {
  float sum = 0.0;
  for (const auto& it : activated_experts_history_) {
    sum += it;
  }
  return sum / activated_experts_history_.size();
}

}  // namespace fcfs_strategy
}  // namespace stragegies