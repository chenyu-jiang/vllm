#include "fcfs.hpp"

#include <algorithm>

namespace stragegies {
namespace fcfs_strategy {

using dependency_graph::AttnNode;
using dependency_graph::AttnNodes;
using dependency_graph::ExpertNode;
using dependency_graph::ExpertNodes;
using dependency_graph::GraphNode;
using dependency_graph::GraphNodePtr;
using dependency_graph::GraphNodes;

FCFSStrategy::FCFSStrategy(const RequestGraphs& graphs,
                           const StrategyConfig& config)
    : Strategy(graphs, config) {
  n_layers_ = config.GetInt("n_layers");
  max_batch_size_ = config.GetInt("max_batch_size");
  max_kv_tokens_ = config.GetInt("max_kv_tokens");
  per_token_latency_slo_ = config_.GetFloat("per_token_latency_slo");
}

ScheduleResult FCFSStrategy::Schedule(
    const std::unordered_map<int, RequestStats>& request_stats,
    const GraphNodes& ready_nodes, float current_time) {
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
    int current_batch_kv_tokens = 0;
    GraphNodes ready_first_layer_attn_nodes;
    for (const auto& node : ready_nodes) {
      if (node->layer_id == 0 && node->Type() == NodeType::kAttn) {
        ready_first_layer_attn_nodes.push_back(node);
      }
    }
    for (const auto& node : ready_first_layer_attn_nodes) {
      if (prev_batch_requests_.find({node->req_id, node->token_index - 1}) !=
          prev_batch_requests_.end()) {
        current_batch_requests_.insert({node->req_id, node->token_index});
        current_batch_kv_tokens +=
            request_stats.at(node->req_id).prompt_len + request_stats.at(node->req_id).GetNumTokensDecoded();
      }
    }
    if ((int)current_batch_requests_.size() < max_batch_size_ &&
        current_batch_kv_tokens < max_kv_tokens_) {
      if (current_layer_ != 0 || current_phase_ != NodeType::kAttn) {
        throw std::runtime_error(
            "ready_first_layer_attn_nodes is not full but current_layer_ is "
            "not 0 or current_phase_ is not kAttn");
      }
      // schedule all first layer Attn requests in FCFS order
      // first sort ready nodes by enqueue time
      GraphNodes sorted_ready_nodes = ready_first_layer_attn_nodes;
      std::sort(sorted_ready_nodes.begin(), sorted_ready_nodes.end(),
                [&](const GraphNodePtr a, const GraphNodePtr b) {
                  return request_stats.at(a->req_id).enqueue_time <
                         request_stats.at(b->req_id).enqueue_time;
                });
      for (const auto& node : sorted_ready_nodes) {
        if (current_batch_requests_.find({node->req_id, node->token_index}) ==
            current_batch_requests_.end()) {
          current_batch_requests_.insert({node->req_id, node->token_index});
          current_batch_kv_tokens +=
              request_stats.at(node->req_id).GetTotalContextLength();
          if ((int)current_batch_requests_.size() == max_batch_size_ ||
              current_batch_kv_tokens >= max_kv_tokens_) {
            break;
          }
        }
      }
    }
    LOG("Scheduling a batch of "
        << current_batch_requests_.size()
        << " requests, total KV size: " << current_batch_kv_tokens << " / "
        << max_kv_tokens_ << " tokens." << std::endl);
    UpdateAvgBatchSize_(current_batch_requests_.size());
  }
  // first test if the current batch is all finished
  // we assume all nodes are DecodeNodes
  GraphNodes current_batch_ready_nodes;
  for (const auto& node : ready_nodes) {
    if (current_batch_requests_.find({node->req_id, node->token_index}) !=
        current_batch_requests_.end()) {
      current_batch_ready_nodes.push_back(node);
    }
  }
  if (current_batch_ready_nodes.empty()) {
    // current batch is all finished
    // advance to the next batch
    ResetBatch_();
    return Schedule(request_stats, ready_nodes, current_time);
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
    ExpertNodes ready_expert_nodes;
    for (const auto& node : current_phase_ready_nodes) {
      ready_expert_nodes.push_back(
          std::dynamic_pointer_cast<ExpertNode>(node));
    }
    if (ready_expert_nodes.empty()) {
      throw std::runtime_error(
          "current_phase_ready_nodes is empty but current_phase_ is "
          "kExpert");
    }
    int expert_id_to_schedule = ready_expert_nodes[0]->expert_id;
    GraphNodes nodes_to_schedule;
    for (const auto& node : ready_expert_nodes) {
      if (node->expert_id == expert_id_to_schedule) {
        nodes_to_schedule.push_back(node);
      }
    }
    activated_experts_per_layer_ += 1;
    return {ModelComponent::kExpert, nodes_to_schedule};
  }
  // schedule all attn nodes together
  return {ModelComponent::kAttnGate, current_phase_ready_nodes};
}

void FCFSStrategy::AdvancePhase_() {
  if (current_phase_ == NodeType::kAttn) {
    current_phase_ = NodeType::kExpert;
  } else {
    UpdateAvgActivatedExperts_(activated_experts_per_layer_);
    activated_experts_per_layer_ = 0;
    current_phase_ = NodeType::kAttn;
    current_layer_ += 1;
  }
}

void FCFSStrategy::ResetBatch_() {
  UpdateAvgActivatedExperts_(activated_experts_per_layer_);
  activated_experts_per_layer_ = 0;
  prev_batch_requests_ = current_batch_requests_;
  current_batch_requests_.clear();
  current_phase_ = NodeType::kAttn;
  current_layer_ = 0;
}

float FCFSStrategy::GetAvgActivatedExperts() const {
  return avg_activated_experts_.first;
}

float FCFSStrategy::GetAvgBatchSize() const { return avg_batch_size_.first; }

void FCFSStrategy::UpdateAvgActivatedExperts_(int n_experts) {
  avg_activated_experts_.first =
      (avg_activated_experts_.first * avg_activated_experts_.second +
       n_experts) /
      (avg_activated_experts_.second + 1);
  avg_activated_experts_.second += 1;
}

void FCFSStrategy::UpdateAvgBatchSize_(int batch_size) {
  avg_batch_size_.first =
      (avg_batch_size_.first * avg_batch_size_.second + batch_size) /
      (avg_batch_size_.second + 1);
  avg_batch_size_.second += 1;
}

}  // namespace fcfs_strategy
}  // namespace stragegies