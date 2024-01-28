#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unordered_set>

#include "dependency_graph.hpp"
#include "schedule_strategies/strategy_factory.hpp"

namespace py = pybind11;

using dependency_graph::BuildGraphs;
using dependency_graph::GraphNodes;
using dependency_graph::NodeType;
using dependency_graph::DecodeNode;
using dependency_graph::PerTokenExpertIds;
using dependency_graph::TokenIds;
using stragegies::ModelComponentNames;
using stragegies::RequestStats;
using stragegies::StrategyConfig;
using stragegies::StrategyFactory;

using RequestStatsDict = std::unordered_map<int, RequestStats>;
using SimulationResult =
    std::tuple<RequestStatsDict, float, float, float, float>;

SimulationResult RunSimulation(
    py::object cost_model,
    const std::vector<PerTokenExpertIds>& expert_selections,
    const std::vector<TokenIds>& context_token_ids,
    const std::vector<TokenIds>& decoded_token_ids, int n_layers,
    int n_experts, int max_batch_size, float per_token_latency_slo,
    std::string schedule_strategy) {
  // first check if the schedule strategy is valid
  if (!StrategyFactory::Has(schedule_strategy)) {
    throw std::runtime_error("Invalid schedule strategy: " +
                             schedule_strategy);
  }
  // build request graphs
  auto request_graphs = BuildGraphs(expert_selections, context_token_ids,
                                    decoded_token_ids, n_layers);
  LOG("Built " << request_graphs.size() << " graphs." << std::endl);
  LOG("Strategy: " << schedule_strategy << std::endl);
  // get the strategy
  StrategyConfig config;
  config.Set("n_layers", n_layers);
  config.Set("n_experts", n_experts);
  config.Set("per_token_latency_slo", per_token_latency_slo);
  auto strategy =
      StrategyFactory::Make(schedule_strategy, request_graphs, config);
  // prepare variables for simulation
  RequestStatsDict request_stats;
  float peak_kv_tokens = 0.0;
  float avg_cost_per_step = 0.0;
  int n_steps = 0;
  float current_time = 0.0;
  int total_tokens_processed = 0;
  for (const auto& graph : request_graphs) {
    request_stats[graph.req_id] = RequestStats(graph.req_id, current_time);
  }
  // run simulation
  while (true) {
    // get ready nodes
    GraphNodes ready_nodes;
    for (const auto& graph : request_graphs) {
      auto graph_ready_nodes = graph.GetFrontier();
      for (const auto& node : graph_ready_nodes) {
        ready_nodes.push_back(node);
      }
    }
    // update peak KV tokens
    std::unordered_set<int> all_active_reqs;
    int all_inflight_kv_tokens = 0;
    for (const auto& node : ready_nodes) {
      if (node->req_id != 0) {
        all_active_reqs.insert(node->req_id);
      }
    }
    for (const auto& req_id : all_active_reqs) {
      all_inflight_kv_tokens +=
          request_graphs[req_id].prompt_token_ids.size() +
          request_stats[req_id].GetNumTokensDecoded();
    }
    peak_kv_tokens = std::max(peak_kv_tokens, (float)all_inflight_kv_tokens);
    // schedule
    auto schedule_result =
        strategy.Schedule(request_stats, ready_nodes, max_batch_size);
    auto& model_component = schedule_result.first;
    auto& scheduled_nodes = schedule_result.second;
    if (scheduled_nodes.empty()) {
      // no more requests to schedule
      break;
    }
    // calculate cost
    int total_context_len = 0;
    for (const auto& node : scheduled_nodes) {
      total_context_len += node->prompt_len + node->token_index;
    }
    int batch_size = scheduled_nodes.size();
    float cost = cost_model
                     .attr("get_cost")(ModelComponentNames.at(model_component),
                                       batch_size, total_context_len)
                     .cast<float>();
    current_time += cost;
    avg_cost_per_step += cost;
    n_steps += 1;
    // release dependency
    for (auto& node : scheduled_nodes) {
      auto& graph = request_graphs[node->req_id];
      graph.Execute(*node);
      if (dynamic_cast<const DecodeNode*>(node) && 
          dynamic_cast<const DecodeNode*>(node)->IsLastNodeForToken()) {
        request_stats[node->req_id].RecordTokenFinish(current_time);
        total_tokens_processed += 1;
      }
    }
  }
  // calculate avg cost per step
  float throughput = total_tokens_processed / current_time * 1000.0;
  avg_cost_per_step = avg_cost_per_step / n_steps;
  return {request_stats, throughput, avg_cost_per_step, peak_kv_tokens,
          strategy.GetAvgActivatedExperts()};
}

PYBIND11_MODULE(simulator, m) {
  m.doc() = "C++ implementation of the simulation algorithm";
  m.def("run_simulation", &RunSimulation, "Run simulation",
        py::arg("cost_model"), py::arg("expert_selections"),
        py::arg("context_token_ids"), py::arg("decoded_token_ids"),
        py::arg("n_layers"), py::arg("n_experts"), py::arg("max_batch_size"),
        py::arg("per_token_latency_slo"), py::arg("schedule_strategy"));
}