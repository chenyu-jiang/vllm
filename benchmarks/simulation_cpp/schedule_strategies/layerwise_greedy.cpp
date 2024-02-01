#include "layerwise_greedy.hpp"

#include <algorithm>
#include <map>

namespace stragegies {
namespace lwg_strategy {

using dependency_graph::AttnNode;
using dependency_graph::AttnNodes;
using dependency_graph::ExpertNode;
using dependency_graph::ExpertNodes;
using dependency_graph::GraphNode;
using dependency_graph::GraphNodePtr;
using dependency_graph::GraphNodes;

using PerLayerBatchingResult =
    std::map<int, std::tuple<std::set<int>, int, int>>;

// layer_id -> req_id -> (expert_ids, time_since_last_token)
using PerLayerPerRequestExpertSelection =
    std::map<int, std::map<int, std::pair<std::set<int>, float>>>;

PerLayerBatchingResult FindBestTokensGreedy(
    PerLayerPerRequestExpertSelection expert_selection_map, int max_batch_size,
    int n_layers, float avg_layer_cost, float per_token_latency_slo,
    float severity_threshold = 0.9) {
  PerLayerBatchingResult best_tokens_by_layer;
  for (const auto& kv : expert_selection_map) {
    int layer_id = kv.first;
    const auto& req_id_to_expert_ids = kv.second;
    std::set<int> req_ids_to_select;
    std::set<int> unselected_req_ids;
    std::set<int> current_active_experts;
    LOG("Layer " << layer_id << " has " << req_id_to_expert_ids.size()
                 << " requests." << std::endl);
    int effective_batch_size =
        std::min(max_batch_size, (int)req_id_to_expert_ids.size());
    // determine the severity for each request
    std::map<int, float> high_priority_req_ids;
    for (const auto& kv : req_id_to_expert_ids) {
      int req_id = kv.first;
      float time_since_last_token = kv.second.second;
      float estimated_latency_lowerbound =
          time_since_last_token + avg_layer_cost * (n_layers - layer_id - 1);
      if (estimated_latency_lowerbound >
          severity_threshold * per_token_latency_slo) {
        high_priority_req_ids[req_id] = estimated_latency_lowerbound;
      }
    }
    int selected_high_prio_reqs = 0;
    // init selection, first with high priority requests
    for (auto it = high_priority_req_ids.rbegin();
         it != high_priority_req_ids.rend(); it++) {
      if ((int)req_ids_to_select.size() < effective_batch_size) {
        req_ids_to_select.insert(it->first);
        for (const auto& expert_id :
             req_id_to_expert_ids.at(it->first).first) {
          current_active_experts.insert(expert_id);
        }
        selected_high_prio_reqs++;
      }
    }
    // then greedily fill the rest
    for (const auto& kv : req_id_to_expert_ids) {
      if (req_ids_to_select.find(kv.first) == req_ids_to_select.end()) {
        unselected_req_ids.insert(kv.first);
      }
    }
    // greedily select the best tokens
    while ((int)req_ids_to_select.size() < effective_batch_size) {
      int min_new_activated_experts = std::numeric_limits<int>::max();
      int best_req_id = -1;
      for (const auto& req_id : unselected_req_ids) {
        int new_activated_experts = 0;
        for (const auto& expert_id : req_id_to_expert_ids.at(req_id).first) {
          if (current_active_experts.find(expert_id) ==
              current_active_experts.end()) {
            new_activated_experts++;
          }
        }
        if (new_activated_experts < min_new_activated_experts) {
          min_new_activated_experts = new_activated_experts;
          best_req_id = req_id;
        }
      }
      // update selection
      req_ids_to_select.insert(best_req_id);
      unselected_req_ids.erase(best_req_id);
      for (const auto& expert_id :
           req_id_to_expert_ids.at(best_req_id).first) {
        current_active_experts.insert(expert_id);
      }
    }
    LOG("Choosing " << effective_batch_size << " from "
                    << req_id_to_expert_ids.size() << " tokens, activated "
                    << current_active_experts.size() << " experts, containing "
                    << selected_high_prio_reqs << " high priority requests."
                    << std::endl);
    best_tokens_by_layer[layer_id] = {req_ids_to_select,
                                      current_active_experts.size(),
                                      selected_high_prio_reqs};
  }
  return best_tokens_by_layer;
}

LayerWiseGreedyStrategy::LayerWiseGreedyStrategy(const RequestGraphs& graphs,
                                                 const StrategyConfig& config)
    : Strategy(graphs, config) {
  n_layers_ = config.GetInt("n_layers");
  max_batch_size_ = config.GetInt("max_batch_size");
  per_token_latency_slo_ = config_.GetFloat("per_token_latency_slo");
  k_experts_per_token_ = config_.GetInt("k_experts_per_token");
}

ScheduleResult LayerWiseGreedyStrategy::Schedule(
    const std::unordered_map<int, RequestStats>& request_stats,
    const GraphNodes& ready_nodes, float current_time) {
  if (ready_nodes.empty()) {
    // no more requests to schedule
    return ScheduleResult(ModelComponent::kAttnGate, {});
  }
  if (current_req_ids_.empty()) {
    PerLayerPerRequestExpertSelection expert_selection_map;
    std::map<int, int> per_request_current_token_node_ids;
    for (const auto& node : ready_nodes) {
      if (node->Type() == NodeType::kAttn) {
        if (per_request_current_token_node_ids.find(node->req_id) !=
            per_request_current_token_node_ids.end()) {
          throw std::runtime_error(
              "Found multiple ready attn nodes for the same request");
        }
        per_request_current_token_node_ids[node->req_id] = node->node_id;
      }
    }
    for (const auto& it : per_request_current_token_node_ids) {
      const auto& nodes_it = graphs_[it.first].GetNodes().begin() + it.second;
      const auto& attn_node = *nodes_it;
      GraphNodes expert_graph_nodes(nodes_it + 1,
                                    nodes_it + k_experts_per_token_ + 1);
      std::set<int> selected_experts;
      for (const auto& node : expert_graph_nodes) {
        selected_experts.insert(
            std::dynamic_pointer_cast<ExpertNode>(node)->expert_id);
      }
      expert_selection_map[attn_node->layer_id][attn_node->req_id] = {
          std::move(selected_experts),
          request_stats.at(attn_node->req_id)
              .GetTimeSinceLastToken(current_time,
                                     /*include_first_token=*/false)};
    }
    LOG("Built expert selection map containing " << expert_selection_map.size()
                                                 << " layers." << std::endl);
    // find avg layer cost
    float avg_layer_cost =
        GetAvgActivatedExperts() * avg_node_latency_[NodeType::kExpert].first +
        avg_node_latency_[NodeType::kAttn].first;
    // greedily find the best tokens to schedule
    const auto& best_req_ids_by_layer =
        FindBestTokensGreedy(expert_selection_map, max_batch_size_, n_layers_,
                             avg_layer_cost, per_token_latency_slo_);
    // first see if any high priority requests are selected
    int selected_layer = -1;
    int bset_n_high_prio_reqs = 0;
    int best_activated_experts = -1;
    for (const auto& kv : best_req_ids_by_layer) {
      int layer_id = kv.first;
      const auto& it = kv.second;
      int n_high_prio_reqs = std::get<2>(it);
      if (n_high_prio_reqs > bset_n_high_prio_reqs) {
        selected_layer = layer_id;
        bset_n_high_prio_reqs = n_high_prio_reqs;
        best_activated_experts = std::get<1>(it);
      }
    }
    if (bset_n_high_prio_reqs == 0) {
      // select the layer with minimum activated experts / batch size
      float best_expert_over_bs = std::numeric_limits<float>::max();
      for (const auto& kv : best_req_ids_by_layer) {
        int layer_id = kv.first;
        const auto& it = kv.second;
        int activated_experts = std::get<1>(it);
        const auto& selected_req_ids = std::get<0>(it);
        float expert_over_bs =
            (float)activated_experts / selected_req_ids.size();
        if (expert_over_bs < best_expert_over_bs) {
          selected_layer = layer_id;
          best_expert_over_bs = expert_over_bs;
          best_activated_experts = activated_experts;
        }
      }
    }
    LOG("Selected "
        << std::get<0>(best_req_ids_by_layer.at(selected_layer)).size()
        << " tokens from layer " << selected_layer << ", activating "
        << best_activated_experts << " experts"
        << ", containing " << bset_n_high_prio_reqs
        << " high priority requests." << std::endl);
    UpdateAvgActivatedExperts_(best_activated_experts);
    current_req_ids_ = std::get<0>(best_req_ids_by_layer.at(selected_layer));
    UpdateAvgBatchSize_(current_req_ids_.size());
    current_layer_id_ = selected_layer;
  }
  // schedule the current layer and request ids
  GraphNodes current_phase_ready_nodes;
  for (const auto& node : ready_nodes) {
    if (node->Type() == current_phase_) {
      if (current_req_ids_.find(node->req_id) != current_req_ids_.end() &&
          node->layer_id == current_layer_id_) {
        current_phase_ready_nodes.push_back(node);
      }
    }
  }
  if (current_phase_ready_nodes.empty()) {
    // advance phase
    AdvancePhase_();
    return Schedule(request_stats, ready_nodes, current_time);
  }
  if (current_phase_ == NodeType::kExpert) {
    // schedule one expert at a time
    ExpertNodes ready_expert_nodes;
    for (const auto& node : current_phase_ready_nodes) {
      ready_expert_nodes.push_back(
          std::dynamic_pointer_cast<ExpertNode>(node));
    }
    int expert_id_to_schedule = ready_expert_nodes[0]->expert_id;
    GraphNodes nodes_to_schedule;
    for (const auto& node : ready_expert_nodes) {
      if (node->expert_id == expert_id_to_schedule) {
        nodes_to_schedule.push_back(node);
      }
    }
    return ScheduleResult(ModelComponent::kExpert, nodes_to_schedule);
  }
  // schedule all attn nodes
  return ScheduleResult(ModelComponent::kAttnGate, current_phase_ready_nodes);
}

void LayerWiseGreedyStrategy::AdvancePhase_() {
  if (current_phase_ == NodeType::kAttn) {
    current_phase_ = NodeType::kExpert;
  } else {
    // done with this layer
    current_phase_ = NodeType::kAttn;
    current_req_ids_.clear();
  }
}

float LayerWiseGreedyStrategy::GetAvgActivatedExperts() const {
  return avg_activated_experts_.first;
}

float LayerWiseGreedyStrategy::GetAvgBatchSize() const {
  return avg_batch_size_.first;
}

void LayerWiseGreedyStrategy::UpdateAvgActivatedExperts_(int n_experts) {
  avg_activated_experts_.first =
      (avg_activated_experts_.first * avg_activated_experts_.second +
       n_experts) /
      (avg_activated_experts_.second + 1);
  avg_activated_experts_.second += 1;
}

void LayerWiseGreedyStrategy::UpdateAvgBatchSize_(int batch_size) {
  avg_batch_size_.first =
      (avg_batch_size_.first * avg_batch_size_.second + batch_size) /
      (avg_batch_size_.second + 1);
  avg_batch_size_.second += 1;
}

}  // namespace lwg_strategy
}  // namespace stragegies