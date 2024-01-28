#include "dependency_graph.hpp"

#include <algorithm>
#include <stdexcept>

namespace dependency_graph {

void GraphNode::AddChild(GraphNode& child) {
  children_.push_back(&child);
}

void GraphNode::AddParent(GraphNode& parent) {
  parents_.push_back(&parent);
}

bool GraphNode::IsReady() const {
  for (const auto& parent : parents_) {
    if (!parent->is_executed_) {
      return false;
    }
  }
  return true;
}

NodeType GraphNode::Type() const { return NodeType::kBase; }

GraphNodes GraphNode::Execute() {
  is_executed_ = true;
  GraphNodes ready_children;
  for (const auto& child : children_) {
    if (child->IsReady()) {
      ready_children.push_back(child);
    }
  }
  return ready_children;
}

NodeType DecodeNode::Type() const { return NodeType::kDecode; }

int DecodeNode::ContextLen() const { return prompt_len + token_index; }

bool DecodeNode::IsLastNodeForToken() const {
  if (!is_executed_) {
    throw std::runtime_error(
        "DecodeNode::IsLastNodeForToken() called before "
        "DecodeNode::Execute()");
  }
  return is_last_node_for_token_;
}

GraphNodes DecodeNode::Execute() {
  GraphNodes ready_children = GraphNode::Execute();
  GraphNodes filtered_children;
  for (const auto& child : ready_children) {
    if (child->Type() == NodeType::kFin || (child->Type() == NodeType::kDecode &&
                                           child->token_index > token_index)) {
      // this is the last node for a token
      is_last_node_for_token_ = true;
    }
    if (child->Type() != NodeType::kFin) {
      filtered_children.push_back(child);
    }
  }
  return filtered_children;
}

NodeType FinNode::Type() const { return NodeType::kFin; }

NodeType AttnNode::Type() const { return NodeType::kAttn; }

NodeType ExpertNode::Type() const { return NodeType::kExpert; }

void RequestGraph::InitFromNodes(
    std::vector<std::vector<GraphNode>>& from_nodes) {
  nodes_.clear();
  for (size_t i = 1; i < from_nodes.size(); i++) {
    for (auto& node : from_nodes[i]) {
      for (auto& parent : from_nodes[i - 1]) {
        node.AddParent(parent);
        parent.AddChild(node);
      }
    }
  }
  for (const auto& nodes_at_layer : from_nodes) {
    for (const auto& node : nodes_at_layer) {
      nodes_.push_back(node);
    }
  }
  // sort nodes by node_id
  std::sort(nodes_.begin(), nodes_.end(),
            [](const GraphNode& a, const GraphNode& b) {
              return a.node_id < b.node_id;
            });
  InitFrontier_();
}

GraphNodes RequestGraph::GetFrontier() const { return frontier_; }

void RequestGraph::Execute(GraphNode& node) {
  if (node.req_id != req_id) {
    throw std::runtime_error(
        "RequestGraph::Execute() called with node from "
        "different request");
  }
  if (!node.IsReady()) {
    throw std::runtime_error(
        "RequestGraph::Execute() called with node that "
        "is not ready");
  }
  std::remove(frontier_.begin(), frontier_.end(), &node);
  GraphNodes ready_children = node.Execute();
  for (const auto& child : ready_children) {
    frontier_.push_back(child);
  }
}

GraphNodes RequestGraph::GetNodes() {
  GraphNodes nodes;
  for (GraphNode& node : nodes_) {
    nodes.push_back(&node);
  }
  return nodes;
}

void RequestGraph::InitFrontier_() {
  frontier_.clear();
  for (auto& node : nodes_) {
    if (node.IsReady()) {
      frontier_.push_back(&node);
    }
  }
}



RequestGraphs BuildGraphs(
    const std::vector<PerTokenExpertIds>& expert_selections,
    const std::vector<TokenIds>& context_token_ids,
    const std::vector<TokenIds>& decoded_token_ids, int n_layers) {
  RequestGraphs graphs;
  // check that all vectors are the same size
  if (expert_selections.size() != context_token_ids.size() ||
      expert_selections.size() != decoded_token_ids.size()) {
    throw std::runtime_error(
        "BuildGraphs() called with vectors of different "
        "sizes");
  }
  for (size_t req_id = 0; req_id < expert_selections.size(); req_id++) {
    auto graph = RequestGraph(req_id, context_token_ids[req_id],
                              decoded_token_ids[req_id]);
    int node_id_counter = 0;
    auto get_next_node_id = [&node_id_counter]() { return node_id_counter++; };
    std::vector<std::vector<GraphNode>> nodes;
    for (size_t token_index = 0; token_index < decoded_token_ids[req_id].size();
         token_index++) {
      for (int layer_id = 0; layer_id < n_layers; layer_id++) {
        const ExpertIds& expert_ids =
            expert_selections[req_id][token_index][layer_id];
        auto attn_node =
            AttnNode(get_next_node_id(), req_id, layer_id,
                     context_token_ids[req_id].size(), token_index);
        nodes.push_back({attn_node});
        // experts
        std::vector<GraphNode> expert_nodes;
        for (int expert_id : expert_ids) {
          auto expert_node =
              ExpertNode(get_next_node_id(), req_id, layer_id, expert_id,
                         context_token_ids[req_id].size(), token_index);
          expert_nodes.push_back(expert_node);
        }
        nodes.push_back(expert_nodes);
      }
    }
    nodes.push_back({FinNode(get_next_node_id(), req_id, n_layers)});
    graph.InitFromNodes(nodes);
    graphs.emplace_back(std::move(graph));
  }
  return graphs;
}

}  // namespace dependency_graph