import os
from typing import List, Set, Tuple, Optional, Callable
from collections import defaultdict

from tqdm import tqdm


class GraphNode:
    def __init__(self, req_id: int, layer_id: int):
        self.req_id: int = req_id
        self.layer_id: int = layer_id
        self.parents: List[GraphNode] = []
        self.children: List[GraphNode] = []
        self.executed: bool = False

    def add_child(self, child_node: "GraphNode") -> None:
        self.children.append(child_node)

    def add_parent(self, parent_node: "GraphNode") -> None:
        self.parents.append(parent_node)

    def is_ready(self) -> bool:
        for parent in self.parents:
            if not parent.executed:
                return False
        return True

    def execute(self) -> List["GraphNode"]:
        self.executed = True
        ready_children = []
        for child in self.children:
            if child.is_ready():
                ready_children.append(child)
        return ready_children


class PrefillNode(GraphNode):
    pass


class DecodeNode(GraphNode):
    def __init__(self,
                 prompt_len: int,
                 token_index: int,
                 req_id: int,
                 layer_id: int):
        super().__init__(req_id, layer_id)
        self.prompt_len = prompt_len
        self.token_index = token_index

    @property
    def context_len(self) -> int:
        return self.prompt_len + self.token_index


class AttnNode(DecodeNode):
    pass


class ExpertNode(DecodeNode):
    def __init__(self,
                 expert_id: int,
                 prompt_len: int,
                 token_index: int,
                 req_id: int,
                 layer_id: int):
        super().__init__(prompt_len, token_index, req_id, layer_id)
        self.expert_id = expert_id


class RequestGraph:
    def __init__(self,
                 req_id: int,
                 prompt_token_ids: List[int],
                 decoded_token_ids: List[int]
                 ):
        self.prompt_token_ids = prompt_token_ids
        self.decoded_token_ids = decoded_token_ids
        self.nodes: List[GraphNode] = []
        self.req_id = req_id
        self.frontier: List[GraphNode] = []

    def init_from_list(self, node_list: List[List[GraphNode]]) -> None:
        # node_list is a list of lists of nodes, where each list of nodes
        # are automatically registered as children of the previous
        # list of nodes.
        last_dep_layer_nodes: List[GraphNode] = []
        for dep_layer in node_list:
            for node in last_dep_layer_nodes:
                for dep_node in dep_layer:
                    node.add_child(dep_node)
                    dep_node.add_parent(node)
            last_dep_layer_nodes = dep_layer
        # add all nodes to self.nodes
        for dep_layer in node_list:
            self.nodes.extend(dep_layer)
        self._init_frontier()

    def _init_frontier(self) -> None:
        self.frontier = []
        for node in self.nodes:
            if node.is_ready():
                self.frontier.append(node)

    def get_frontier(self,
                     filter: Optional[Callable[[GraphNode], bool]] = None
                     ) -> List[GraphNode]:
        if filter is None:
            return self.frontier.copy()
        return [node for node in self.frontier if filter(node)]

    def execute(self, node: GraphNode) -> None:
        if node.req_id != self.req_id:
            raise ValueError("Node with request id {} cannot be executed "
                             "in request graph with id {}"
                             .format(node.req_id, self.req_id))
        if not node.is_ready():
            raise ValueError(f"Node {node} is not ready to be executed")
        ready_children = node.execute()
        for child in ready_children:
            self.frontier.append(child)

def build_graph_from_dataset(dataset_dir):
    # TODO: hard code file names for now
    # parse test_dump_expert_ids.tsv
    n_layers = 0
    n_experts = 0
    # token_id -> layer_id -> expert_ids
    token_id_to_experts = defaultdict(lambda: defaultdict(int))
    with open(os.path.join(dataset_dir, "test_dump_expert_ids.tsv"), "r") as f:
        f.readline() # skip header
        for line in f:
            token_id, layer_id, expert_ids = line.strip().split("\t")
            expert_ids = expert_ids.split(",")
            token_id = int(token_id)
            layer_id = int(layer_id)
            n_layers = max(n_layers, layer_id + 1)
            n_experts = max(n_experts, max([int(expert_id)
                                            for expert_id in expert_ids]) + 1)
            expert_ids = [int(expert_id) for expert_id in expert_ids]
            token_id_to_experts[token_id][layer_id] = sorted(expert_ids)
    # parse test_dump_token_ids.tsv
    token_id_to_contexts = []
    token_id_to_output_token = {}
    with open(os.path.join(dataset_dir, "test_dump_token_ids.tsv"), "r") as f:
        f.readline() # skip header
        for line in f:
            token_id, context, output_token = line.strip().split("\t")
            token_id = int(token_id)
            context = [int(token) for token in context.split(",")]
            output_token = int(output_token)
            token_id_to_contexts.append((token_id, context))
            token_id_to_output_token[token_id] = output_token
    # organize tokens into requests
    unique_sequences: List[Tuple[Tuple[int], Tuple[int]]] = []

    # def _list_starts_with(l1, l2):
    #     return len(l1) >= len(l2) and l1[:len(l2)] == l2

    # def _get_full_context(recorded_token_ids, orig_context):
    #     return list(orig_context) + [token_id_to_output_token[i] for i in recorded_token_ids]

    for token_id, context in token_id_to_contexts:
        for seq_id, (recorded_token_ids, full_context, orig_context) in enumerate(unique_sequences):
            if context == full_context:
                # same request
                new_recorded_token_ids = recorded_token_ids + [token_id]
                unique_sequences[seq_id] = (new_recorded_token_ids,
                                            full_context + [token_id_to_output_token[token_id]],
                                            orig_context)
                break
        else:
            # new request
            unique_sequences.append(([token_id], context + [token_id_to_output_token[token_id]], context))
    
    print(f"Found {len(unique_sequences)} unique sequences.")

    request_graphs = []
    for req_id, (token_ids, _, orig_context) in enumerate(unique_sequences):
        decoded_token_ids = [token_id_to_output_token[token_id]
                             for token_id in token_ids]
        graph = RequestGraph(req_id, orig_context, decoded_token_ids)
        # build graph
        # TODO: we ignore prefill nodes for now
        graph_nodes = []
        for token_index, token_id in enumerate(token_ids):
            for layer_id in range(n_layers):
                expert_ids = token_id_to_experts[token_id][layer_id]
                # attn
                attn_node = AttnNode(prompt_len=len(orig_context),
                                     token_index=token_index,
                                     req_id=req_id,
                                     layer_id=layer_id)
                graph_nodes.append([attn_node])
                # experts
                expert_nodes = []
                for expert_id in expert_ids:
                    expert_node = ExpertNode(expert_id=expert_id,
                                             prompt_len=len(orig_context),
                                             token_index=token_index,
                                             req_id=req_id,
                                             layer_id=layer_id)
                    expert_nodes.append(expert_node)
                graph_nodes.append(expert_nodes)
        graph.init_from_list(graph_nodes)
        request_graphs.append(graph)
    return request_graphs