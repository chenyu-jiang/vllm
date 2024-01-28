#pragma once

#include <string>
#include <unordered_map>

#include "../dependency_graph.hpp"

namespace stragegies {

using dependency_graph::GraphNodes;
using dependency_graph::RequestGraphs;

enum class ModelComponent {
  kAttnGate,
  kExpert,
};

extern const std::unordered_map<ModelComponent, std::string> ModelComponentNames;

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
  RequestStats(int req_id=0, float enqueue_time = 0.0)
      : req_id(req_id), enqueue_time(enqueue_time) {}

  void RecordTokenFinish(float finish_time);
  float GetFirstTokenlatency() const;
  float GetAvgLatency(bool include_first_token) const;
  std::vector<float> GetPerTokenLatencies() const;
  float GetTimeSinceLastToken(float current_time);
  int GetNumTokensDecoded() const;

  int req_id;
  float enqueue_time;

 protected:
  std::vector<float> per_token_finish_times_;
};

class Strategy {
 public:
  Strategy(const RequestGraphs& graphs, const StrategyConfig& config)
      : graphs_(graphs), config_(config) {}
  virtual ScheduleResult Schedule(
      const std::unordered_map<int, RequestStats>& request_stats,
      const GraphNodes& ready_nodes, int max_batch_size = -1);

  virtual float GetAvgActivatedExperts() const;

 protected:
  const RequestGraphs& graphs_;
  const StrategyConfig config_;
};

}  // namespace stragegies