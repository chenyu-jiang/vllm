#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "../dependency_graph.hpp"

namespace stragegies {

using dependency_graph::GraphNodes;
using dependency_graph::NodeType;
using dependency_graph::RequestGraphs;

enum class ModelComponent {
  kAttnGate,
  kExpert,
};

extern const std::unordered_map<ModelComponent, std::string>
    ModelComponentNames;

using ScheduleResult = std::pair<ModelComponent, GraphNodes>;

class StrategyConfig {
 public:
  void Set(const std::string& key, int value);
  void Set(const std::string& key, float value);
  void Set(const std::string& key, const std::string& value);
  int GetInt(const std::string& key) const;
  float GetFloat(const std::string& key) const;
  std::string GetString(const std::string& key) const;

 protected:
  std::unordered_map<std::string, int> int_config_;
  std::unordered_map<std::string, float> float_config_;
  std::unordered_map<std::string, std::string> str_config_;
};

class RequestStats {
 public:
  RequestStats(int req_id = 0, float enqueue_time = 0.0, int prompt_len = 0,
               int output_len = 0);

  void RecordTokenFinish(float finish_time);
  float GetFirstTokenlatency() const;
  float GetAvgLatency(bool include_first_token) const;
  std::vector<float> GetPerTokenLatencies() const;
  float GetTimeSinceLastToken(float current_time,
                              bool include_first_token) const;
  int GetNumTokensDecoded() const;
  int GetTotalContextLength() const;

  int req_id;
  int prompt_len;
  int output_len;
  float enqueue_time;
  std::vector<float> per_token_finish_times_;

 protected:
};

class Strategy {
 public:
  Strategy(const RequestGraphs& graphs, const StrategyConfig& config)
      : graphs_(graphs), config_(config) {}
  virtual ScheduleResult Schedule(
      const std::unordered_map<int, RequestStats>& request_stats,
      const GraphNodes& ready_nodes, float current_time);

  virtual float GetAvgActivatedExperts() const;
  virtual float GetAvgBatchSize() const;

  void RecordNodeLatency(const NodeType& node_type, float latency);

 protected:
  const RequestGraphs& graphs_;
  const StrategyConfig config_;
  std::unordered_map<NodeType, std::pair<float, int>> avg_node_latency_;
};

using StrategyPtr = std::shared_ptr<Strategy>;

}  // namespace stragegies