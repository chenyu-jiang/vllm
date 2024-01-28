#pragma once

#include <vector>


#define ENABLE_LOG

#ifdef ENABLE_LOG
#  include <iostream>
#  define LOG(x) std::cout << x
#else
#  define LOG(x) (void) 0
#endif

namespace dependency_graph {

using TokenId = int;
using ExpertId = int;
using TokenIds = std::vector<int>;
using ExpertIds = std::vector<int>;
using PerLayerExpertIds = std::vector<ExpertIds>;
using PerTokenExpertIds = std::vector<PerLayerExpertIds>;

class GraphNode;
using GraphNodes = std::vector<GraphNode*>;

class RequestGraph;
using RequestGraphs = std::vector<RequestGraph>;

enum class NodeType {
  kBase,
  kDecode,
  kAttn,
  kExpert,
  kFin,
};

class GraphNode {
 public:
  GraphNode(int node_id = 0, int req_id = 0, int layer_id = 0,
            int prompt_len = 0, int token_index = -1)
      : node_id(node_id),
        req_id(req_id),
        layer_id(layer_id),
        prompt_len(prompt_len),
        token_index(token_index) {}

  void AddChild(GraphNode& child);
  void AddParent(GraphNode& parent);
  bool IsReady() const;

  virtual NodeType Type() const;
  virtual GraphNodes Execute();

  int node_id;
  int req_id;
  int layer_id;
  int prompt_len;
  int token_index;

 protected:
  GraphNodes parents_;
  GraphNodes children_;
  bool is_executed_ = false;
};

class DecodeNode : public GraphNode {
 public:
  DecodeNode(int node_id = 0, int req_id = 0, int layer_id = 0,
             int prompt_len = 0, int token_index = 0)
      : GraphNode(node_id, req_id, layer_id, prompt_len, token_index) {}

  int ContextLen() const;
  bool IsLastNodeForToken() const;

  virtual NodeType Type() const override;
  GraphNodes Execute() override;

 protected:
  bool is_last_node_for_token_ = false;
};

class FinNode : public GraphNode {
 public:
  FinNode(int node_id = 0, int req_id = 0, int layer_id = 0)
      : GraphNode(node_id, req_id, layer_id) {}

  NodeType Type() const override;
};

class AttnNode : public DecodeNode {
 public:
  AttnNode(int node_id = 0, int req_id = 0, int layer_id = 0,
           int prompt_len = 0, int token_index = 0)
      : DecodeNode(node_id, req_id, layer_id, prompt_len, token_index) {}

  NodeType Type() const override;
};

class ExpertNode : public DecodeNode {
 public:
  ExpertNode(int node_id = 0, int req_id = 0, int layer_id = 0,
             int expert_id = 0, int prompt_len = 0, int token_index = 0)
      : DecodeNode(node_id, req_id, layer_id, prompt_len, token_index),
        expert_id(expert_id) {}

  NodeType Type() const override;

  int expert_id;
};

class RequestGraph {
 public:
  RequestGraph(int req_id, const TokenIds& prompt_token_ids,
               const TokenIds& decoded_token_ids)
      : req_id(req_id),
        prompt_token_ids(prompt_token_ids),
        decoded_token_ids(decoded_token_ids) {}

  int req_id;
  TokenIds prompt_token_ids;
  TokenIds decoded_token_ids;

  void InitFromNodes(
      std::vector<std::vector<GraphNode>>& from_nodes);
  GraphNodes GetFrontier() const;
  void Execute(GraphNode& node);
  GraphNodes GetNodes();

 protected:
  void InitFrontier_();
  GraphNodes frontier_;
  std::vector<GraphNode> nodes_;
};

RequestGraphs BuildGraphs(
    const std::vector<PerTokenExpertIds>& expert_selections,
    const std::vector<TokenIds>& context_token_ids,
    const std::vector<TokenIds>& decoded_token_ids, int n_layers);

}  // namespace dependency_graph