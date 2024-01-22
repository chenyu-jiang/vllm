import os
from typing import List, Tuple, Optional, Callable
from collections import defaultdict

import pickle
import tqdm

class GraphNode:
    def __init__(self, node_id: int = 0,
                 req_id: int = 0,
                 layer_id: int = 0):
        self.node_id: int = node_id
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

    def to_json(self) -> dict:
        return {
            "req_id": self.req_id,
            "layer_id": self.layer_id,
            "executed": self.executed,
            "parents": [parent.node_id for parent in self.parents],
            "children": [child.node_id for child in self.children],
        }

    def update_from_json(self, json_dict: dict, nodes: List["GraphNode"]) -> "GraphNode":
        self.node_id = json_dict["node_id"]
        self.req_id = json_dict["req_id"]
        self.layer_id = json_dict["layer_id"]
        self.executed = json_dict["executed"]
        self.parents = [nodes[parent_id] for parent_id in json_dict["parents"]]
        self.children = [nodes[child_id] for child_id in json_dict["children"]]


class PrefillNode(GraphNode):
    pass


class DecodeNode(GraphNode):
    def __init__(self,
                 node_id: int = 0,
                 req_id: int = 0,
                 layer_id: int = 0,
                 prompt_len: int = 0,
                 token_index: int = 0):
        super().__init__(node_id, req_id, layer_id)
        self.prompt_len = prompt_len
        self.token_index = token_index
        self._is_last_node_for_token = False

    @property
    def context_len(self) -> int:
        return self.prompt_len + self.token_index

    def is_last_node_for_token(self) -> bool:
        if not self.executed:
            raise ValueError("Node {} is not executed yet".format(self))
        return self._is_last_node_for_token

    def execute(self) -> List[GraphNode]:
        ready_children = super().execute()
        for child in ready_children:
            if (isinstance(child, PrefillNode) or
                isinstance(child, FinNode) or
                (isinstance(child, DecodeNode) and
                 child.token_index > self.token_index)):
                # this is the last node for a token
                self._is_last_node_for_token = True
        return [child for child in ready_children
                if not isinstance(child, FinNode)]

    def to_json(self) -> dict:
        json_dict = super().to_json()
        json_dict["prompt_len"] = self.prompt_len
        json_dict["token_index"] = self.token_index
        json_dict["is_last_node_for_token"] = self._is_last_node_for_token
        return json_dict

    def update_from_json(self, json_dict: dict, nodes: List["GraphNode"]):
        super().update_from_json(json_dict, nodes)
        self.prompt_len = json_dict["prompt_len"]
        self.token_index = json_dict["token_index"]
        self._is_last_node_for_token = json_dict["is_last_node_for_token"]

class FinNode(GraphNode):
    pass


class AttnNode(DecodeNode):
    pass


class ExpertNode(DecodeNode):
    def __init__(self,
                 node_id: int = 0,
                 req_id: int = 0,
                 layer_id: int = 0,
                 expert_id: int = 0,
                 prompt_len: int = 0,
                 token_index: int = 0):
        super().__init__(node_id, req_id, layer_id, prompt_len, token_index)
        self.expert_id = expert_id

    def to_json(self) -> dict:
        json_dict = super().to_json()
        json_dict["expert_id"] = self.expert_id
        return json_dict

    def update_from_json(self, json_dict: dict, nodes: List["GraphNode"]) -> "GraphNode":
        super().update_from_json(json_dict, nodes)
        self.expert_id = json_dict["expert_id"]


_NAME_TO_NODE_TYPE = {
    "PrefillNode": PrefillNode,
    "DecodeNode": DecodeNode,
    "AttnNode": AttnNode,
    "ExpertNode": ExpertNode,
}

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
        node_list_flattened = sorted([node for dep_layer in node_list
                                      for node in dep_layer],
                                      key=lambda node: node.node_id)
        for node in node_list_flattened:
            if not node.node_id == len(self.nodes):
                import code
                code.interact(local=locals())
            assert node.node_id == len(self.nodes)
            self.nodes.append(node)
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
        self.frontier.remove(node)
        ready_children = node.execute()
        for child in ready_children:
            self.frontier.append(child)

    def to_json(self) -> dict:
        return {
            "req_id": self.req_id,
            "prompt_token_ids": self.prompt_token_ids,
            "decoded_token_ids": self.decoded_token_ids,
            "nodes": [(type(node).__name__, node.to_json()) for node in self.nodes],
        }

    @classmethod
    def from_json(cls, json_dict: dict) -> "RequestGraph":
        graph = cls(json_dict["req_id"],
                    json_dict["prompt_token_ids"],
                    json_dict["decoded_token_ids"])
        # init empty nodes array
        graph.nodes = [
            _NAME_TO_NODE_TYPE[node_type] for node_type, _ in json_dict["nodes"]
        ]
        # load nodes from json
        for json_dict_id, (_, node_json) in enumerate(json_dict["nodes"]):
            assert json_dict_id == node_json["node_id"]
            node = graph.nodes[json_dict_id]
            node.update_from_json(node_json, graph.nodes)
        graph._init_frontier()
        return graph


def build_graph_from_dataset(dataset_dir: str,
                             max_samples: int,
                             max_decoded_tokens: int = None,
                             max_layers: int = None,
                            ) -> Tuple[List[RequestGraph], int]:
    # TODO: hard code file names for now
    # parse test_dump_expert_ids.tsv
    n_layers = 0
    n_experts = 0
    # token_id -> layer_id -> expert_ids
    token_id_to_experts = defaultdict(lambda: defaultdict(int))

    if not os.path.exists(os.path.join(dataset_dir, "dataset.pkl")):
        expert_ids_fn = os.path.join(dataset_dir, "test_dump_expert_ids.tsv")
        with tqdm.tqdm(total=os.path.getsize(expert_ids_fn),
                    desc="Parsing expert ids") as pbar:
            with open(expert_ids_fn, "r") as f:
                l = f.readline() # skip header
                pbar.update(len(l))
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
                    pbar.update(len(line))
        # parse test_dump_token_ids.tsv
        token_id_to_contexts = []
        token_id_to_output_token = {}
        token_ids_fn = os.path.join(dataset_dir, "test_dump_token_ids.tsv")
        with tqdm.tqdm(total=os.path.getsize(token_ids_fn),
                    desc="Parsing token ids") as pbar:
            with open(token_ids_fn, "r") as f:
                l = f.readline() # skip header
                pbar.update(len(l))
                for line in f:
                    token_id, context, output_token = line.strip().split("\t")
                    token_id = int(token_id)
                    context = [int(token) for token in context.split(",")]
                    output_token = int(output_token)
                    token_id_to_contexts.append((token_id, context))
                    token_id_to_output_token[token_id] = output_token
                    pbar.update(len(line))
        # organize tokens into requests
        unique_sequences: List[Tuple[Tuple[int], Tuple[int]]] = []

        for token_id, context in tqdm.tqdm(token_id_to_contexts,
                                        desc="Organizing tokens into requests"):
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
        with open(os.path.join(dataset_dir, "dataset.pkl"), "wb") as f:
            pickle.dump((unique_sequences, token_id_to_output_token, dict(token_id_to_experts), n_layers), f)
    else:
        with open(os.path.join(dataset_dir, "dataset.pkl"), "rb") as f:
            unique_sequences, token_id_to_output_token, token_id_to_experts, n_layers = pickle.load(f)

    unique_sequences = unique_sequences[:max_samples]
    if max_layers:
        n_layers = min(n_layers, max_layers)

    request_graphs = []
    for req_id, (token_ids, _, orig_context) in tqdm.tqdm(enumerate(unique_sequences),
                                                          total=len(unique_sequences),
                                                          desc="Building request graphs"):
        decoded_token_ids = [token_id_to_output_token[token_id]
                             for token_id in token_ids]
        if max_decoded_tokens:
            decoded_token_ids = decoded_token_ids[:max_decoded_tokens]
        graph = RequestGraph(req_id, orig_context, decoded_token_ids)
        node_id_counter = 0
        def _get_next_node_id():
            nonlocal node_id_counter
            node_id_counter += 1
            return node_id_counter - 1
        # build graph
        # TODO: we ignore prefill nodes for now
        graph_nodes = []
        for token_index, token_id in enumerate(token_ids):
            for layer_id in range(n_layers):
                expert_ids = token_id_to_experts[token_id][layer_id]
                # attn
                attn_node = AttnNode(node_id=_get_next_node_id(),
                                     req_id=req_id,
                                     layer_id=layer_id,
                                     prompt_len=len(orig_context),
                                     token_index=token_index)
                graph_nodes.append([attn_node])
                # experts
                expert_nodes = []
                for expert_id in expert_ids:
                    expert_node = ExpertNode(node_id=_get_next_node_id(),
                                             req_id=req_id,
                                             layer_id=layer_id,
                                             expert_id=expert_id,
                                             prompt_len=len(orig_context),
                                             token_index=token_index)
                    expert_nodes.append(expert_node)
                graph_nodes.append(expert_nodes)
            if max_decoded_tokens and token_index == max_decoded_tokens - 1:
                break
        graph_nodes.append([FinNode(node_id=_get_next_node_id(), req_id=req_id, layer_id=n_layers)])
        graph.init_from_list(graph_nodes)
        request_graphs.append(graph)
    return request_graphs, n_layers, len(list(token_id_to_experts.values())[0][0])