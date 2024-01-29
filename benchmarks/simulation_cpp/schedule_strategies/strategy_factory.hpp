#pragma once

#include <functional>
#include <unordered_map>

#include "../dependency_graph.hpp"
#include "strategy.hpp"

#define REGISTER_STRATEGY(StrategyName, ClassName)      \
  StrategyMaker __maker_##StrategyName =                \
      [](const RequestGraphs& graphs,                   \
         const StrategyConfig& config) -> StrategyPtr { \
    return StrategyPtr(new ClassName(graphs, config));  \
  };                                                    \
  static std::string __reg_##Strategy__COUNTER__ =      \
      StrategyFactory::Register(#StrategyName, __maker_##StrategyName);

namespace stragegies {

using dependency_graph::RequestGraphs;

using StrategyMaker =
    std::function<StrategyPtr(const RequestGraphs&, const StrategyConfig&)>;

class StrategyFactory {
  // Singleton class managing all registered strategies.
 public:
  StrategyFactory(StrategyFactory const&) = delete;
  void operator=(StrategyFactory const&) = delete;
  static StrategyFactory& Get();

  static std::string Register(const std::string& name, StrategyMaker maker);

  static StrategyPtr Make(const std::string& name, const RequestGraphs& graphs,
                          const StrategyConfig& config);

  static bool Has(const std::string& name);

 protected:
  StrategyFactory() = default;
  std::unordered_map<std::string, StrategyMaker> makers_;
};

}  // namespace stragegies
