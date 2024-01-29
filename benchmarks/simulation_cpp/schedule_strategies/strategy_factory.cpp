#include "strategy_factory.hpp"

namespace stragegies {

StrategyFactory& StrategyFactory::Get() {
  static StrategyFactory instance;
  return instance;
}

std::string StrategyFactory::Register(const std::string& name,
                                      StrategyMaker maker) {
  auto& factory = Get();
  factory.makers_[name] = maker;
  return name;
}

StrategyPtr StrategyFactory::Make(const std::string& name,
                                  const RequestGraphs& graphs,
                                  const StrategyConfig& config) {
  auto& factory = Get();
  return factory.makers_[name](graphs, config);
}

bool StrategyFactory::Has(const std::string& name) {
  auto& factory = Get();
  return factory.makers_.find(name) != factory.makers_.end();
}

}  // namespace stragegies