#include "strategy.hpp"

namespace stragegies {

const std::unordered_map<ModelComponent, std::string> ModelComponentNames = {
    {ModelComponent::kAttnGate, "attn_gate"},
    {ModelComponent::kExpert, "expert"},
};

void StrategyConfig::Set(const std::string& key, int value) {
  int_config_[key] = value;
}

void StrategyConfig::Set(const std::string& key, float value) {
  float_config_[key] = value;
}

void StrategyConfig::Set(const std::string& key, const std::string& value) {
  str_config_[key] = value;
}

int StrategyConfig::GetInt(const std::string& key) const {
  if (int_config_.find(key) == int_config_.end()) {
    throw std::runtime_error("Key " + key + " not found in config.");
  }
  return int_config_.at(key);
}

float StrategyConfig::GetFloat(const std::string& key) const {
  if (float_config_.find(key) == float_config_.end()) {
    throw std::runtime_error("Key " + key + " not found in config.");
  }
  return float_config_.at(key);
}

std::string StrategyConfig::GetString(const std::string& key) const {
  if (str_config_.find(key) == str_config_.end()) {
    throw std::runtime_error("Key " + key + " not found in config.");
  }
  return str_config_.at(key);
}

void RequestStats::RecordTokenFinish(float finish_time) {
  per_token_finish_times_.push_back(finish_time);
}

float RequestStats::GetFirstTokenlatency() const {
  if (per_token_finish_times_.empty()) {
    throw std::runtime_error("No tokens finished.");
  }
  return per_token_finish_times_[0] - enqueue_time;
}

float RequestStats::GetAvgLatency(bool include_first_token) const {
  if (per_token_finish_times_.empty()) {
    throw std::runtime_error("No tokens finished.");
  }
  float sum = 0.0;
  int n_tokens =
      per_token_finish_times_.size() - (include_first_token ? 0 : 1);
  for (size_t i = 1; i < per_token_finish_times_.size(); ++i) {
    sum += per_token_finish_times_[i] - per_token_finish_times_[i - 1];
  }
  if (include_first_token) {
    sum += GetFirstTokenlatency();
  }
  return sum / n_tokens;
}

std::vector<float> RequestStats::GetPerTokenLatencies() const {
  if (per_token_finish_times_.empty()) {
    throw std::runtime_error("No tokens finished.");
  }
  std::vector<float> latencies;
  latencies.push_back(GetFirstTokenlatency());
  for (size_t i = 1; i < per_token_finish_times_.size(); ++i) {
    latencies.push_back(per_token_finish_times_[i] -
                        per_token_finish_times_[i - 1]);
  }
  return latencies;
}

float RequestStats::GetTimeSinceLastToken(float current_time) {
  if (per_token_finish_times_.empty()) {
    return current_time - enqueue_time;
  }
  return current_time - per_token_finish_times_.back();
}

int RequestStats::GetNumTokensDecoded() const {
  return per_token_finish_times_.size();
}

ScheduleResult Strategy::Schedule(
    const std::unordered_map<int, RequestStats>& request_stats,
    const GraphNodes& ready_nodes, int max_batch_size) {
  throw std::runtime_error("Schedule not implemented in Strategy Base.");
}

float Strategy::GetAvgActivatedExperts() const {
  throw std::runtime_error(
      "GetAvgActivatedExperts not implemented in Strategy Base.");
}

}  // namespace stragegies