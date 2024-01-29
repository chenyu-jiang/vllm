#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <unordered_set>

#include "dependency_graph.hpp"
#include "schedule_strategies/strategy_factory.hpp"

// #define BENCHMARK

#ifdef BENCHMARK
#define DEF_BENCH_NAMED(name)                       \
  auto name##_accum = std::chrono::microseconds(0); \
  auto name##_cnter = 0;
#define START_BENCH(name) auto name##_begin = std::chrono::steady_clock::now();
#define END_BENCH(name)                                                  \
  auto name##_end = std::chrono::steady_clock::now();                    \
  name##_accum += std::chrono::duration_cast<std::chrono::microseconds>( \
      name##_end - name##_begin);                                        \
  name##_cnter++;
#define PRINT_BENCH(name)                       \
  printf("%s took %f seconds to run.\n", #name, \
         (double)name##_accum.count() / 1000000 / name##_cnter);
#else
#define DEF_BENCH_NAMED(name) (void)0
#define START_BENCH(name) (void)0
#define END_BENCH(name) (void)0
#define PRINT_BENCH(name) (void)0
#endif

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

void printProgress(double percentage) {
  static double last_percentage = 0;
  if (percentage - last_percentage < 0.01) {
    return;
  }
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
  last_percentage = percentage;
}

namespace py = pybind11;

using dependency_graph::BuildGraphs;
using dependency_graph::DecodeNode;
using dependency_graph::GraphNodes;
using dependency_graph::NodeType;
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
    int n_experts, int k_experts_per_token,
    int max_batch_size, float per_token_latency_slo,
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
  config.Set("k_experts_per_token", k_experts_per_token);
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
  int requests_finished = 0;
  DEF_BENCH_NAMED(Total);
  DEF_BENCH_NAMED(Schedule);
  DEF_BENCH_NAMED(GetReadyNode);
  DEF_BENCH_NAMED(CostModel);
  // run simulation
  START_BENCH(Total);
  while (true) {
    // get ready nodes
    START_BENCH(GetReadyNode);
    GraphNodes ready_nodes;
    ready_nodes.reserve(request_graphs.size() * n_layers * 256);
    for (const auto& graph : request_graphs) {
      const auto& graph_ready_nodes = graph.GetFrontier();
      ready_nodes.insert(ready_nodes.end(), graph_ready_nodes.begin(),
                         graph_ready_nodes.end());
    }
    END_BENCH(GetReadyNode);
    // update peak KV tokens
    std::unordered_set<int> all_active_reqs;
    int all_inflight_kv_tokens = 0;
    for (const auto& node : ready_nodes) {
      if (node->layer_id != 0) {
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
    LOG("Ready nodes: " << ready_nodes.size() << std::endl);
    START_BENCH(Schedule);
    auto schedule_result =
        strategy->Schedule(request_stats, ready_nodes, max_batch_size);
    END_BENCH(Schedule);
    auto& model_component = schedule_result.first;
    auto& scheduled_nodes = schedule_result.second;
    if (scheduled_nodes.empty()) {
      // no more requests to schedule
      break;
    }
    LOG("Scheduled " << scheduled_nodes.size() << " nodes." << std::endl);
    // calculate cost
    int total_context_len = 0;
    for (const auto& node : scheduled_nodes) {
      total_context_len += node->prompt_len + node->token_index;
    }
    int batch_size = scheduled_nodes.size();
    START_BENCH(CostModel);
    float cost = cost_model
                     .attr("get_cost")(ModelComponentNames.at(model_component),
                                       batch_size, total_context_len)
                     .cast<float>();
    END_BENCH(CostModel);
    current_time += cost;
    avg_cost_per_step += cost;
    n_steps += 1;
    LOG("Cost: " << cost << ", Current time: " << current_time << std::endl);
    // release dependency
    for (auto& node : scheduled_nodes) {
      auto& graph = request_graphs[node->req_id];
      graph.Execute(node);
      if (std::dynamic_pointer_cast<DecodeNode>(node) &&
          std::dynamic_pointer_cast<DecodeNode>(node)->IsLastNodeForToken()) {
        request_stats[node->req_id].RecordTokenFinish(current_time);
        total_tokens_processed += 1;
        LOG("Request " << node->req_id << " finished "
                       << request_stats[node->req_id].GetNumTokensDecoded()
                       << " tokens." << std::endl);
        if (request_stats[node->req_id].GetNumTokensDecoded() ==
            (int)graph.decoded_token_ids.size()) {
          requests_finished++;
          // all tokens are decoded
          printProgress((double)requests_finished / request_graphs.size());
        }
      }
    }
  }
  END_BENCH(Total);
  // calculate avg cost per step
  float throughput = total_tokens_processed / current_time * 1000.0;
  avg_cost_per_step = avg_cost_per_step / n_steps;
  PRINT_BENCH(Total);
  PRINT_BENCH(Schedule);
  PRINT_BENCH(GetReadyNode);
  PRINT_BENCH(CostModel);
  return {request_stats, throughput, avg_cost_per_step, peak_kv_tokens,
          strategy->GetAvgActivatedExperts()};
}

PYBIND11_MODULE(simulator, m) {
  m.doc() = "C++ implementation of the simulation algorithm";
  m.def("run_simulation", &RunSimulation, "Run simulation",
        py::arg("cost_model"),
        py::arg("expert_selections"),
        py::arg("context_token_ids"),
        py::arg("decoded_token_ids"),
        py::arg("n_layers"),
        py::arg("n_experts"),
        py::arg("k_experts_per_token"),
        py::arg("max_batch_size"),
        py::arg("per_token_latency_slo"),
        py::arg("schedule_strategy"));

  py::class_<RequestStats>(m, "RequestStats")
      .def(py::init<int, float>())
      .def_readwrite("req_id", &RequestStats::req_id)
      .def_readwrite("enqueue_time", &RequestStats::enqueue_time)
      .def_readwrite("per_token_finish_times",
                     &RequestStats::per_token_finish_times_)
      .def("record_token_finish", &RequestStats::RecordTokenFinish)
      .def("get_first_token_latency", &RequestStats::GetFirstTokenlatency)
      .def("get_avg_latency", &RequestStats::GetAvgLatency)
      .def("get_per_token_latencies", &RequestStats::GetPerTokenLatencies)
      .def("get_time_since_last_token", &RequestStats::GetTimeSinceLastToken)
      .def("get_num_tokens_decoded", &RequestStats::GetNumTokensDecoded);
}