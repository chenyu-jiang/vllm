from typing import Dict, List, Set, Optional, Tuple
import time


from .base import ScheduleStrategy, register_schedule_strategy
from ..simulator import RequestStats
from ..dependency_graph import GraphNode, AttnNode, ExpertNode
from ..logger import logger
from ..dependency_graph import RequestGraph
from vllm.transformers_utils.cost_model import ModelComponent

import gurobipy as gp

_EARLY_STOP_TIME_LIMIT = 2
_PER_BATCH_EXPERT_EARLY_STOP = 190

def _early_stop_cb(model, where):
    if where == gp.GRB.Callback.MIPNODE:
        # Get model objective
        obj = model.cbGet(gp.GRB.Callback.MIPNODE_OBJBST)

        # Has objective changed?
        if abs(obj - model._cur_obj) > 1e-8:
            # If so, update incumbent and time
            model._cur_obj = obj
            model._time = time.time()

    # Terminate if objective has not improved in 2s
    if time.time() - model._time > _EARLY_STOP_TIME_LIMIT and \
        model._cur_obj < _PER_BATCH_EXPERT_EARLY_STOP:
        model.terminate()

def _c_act_exps(nodes: List[GraphNode]):
    activated_experts = set()
    for node in nodes:
        if isinstance(node, ExpertNode):
            activated_experts.add((node.layer_id, node.expert_id))
    return activated_experts

def _calculate_activated_experts(current_token_nodes: List[GraphNode], seq_indices: Set[int]):
    activated_experts = set()
    for i in seq_indices:
        for node in current_token_nodes[i]:
            if isinstance(node, ExpertNode):
                activated_experts.add((node.layer_id, node.expert_id))
    return len(activated_experts)

def find_best_tokens_local_search(current_token_nodes_dict: Dict[int, List[List[GraphNode]]],
                                  max_batch_size: int) -> List[int]:
    results = {}
    for layer_id, current_token_nodes in current_token_nodes_dict.items():
        effective_max_batch_size = min(max_batch_size, len(current_token_nodes))
        selected_indices = set(range(effective_max_batch_size))
        while True:
            best_score = _calculate_activated_experts(current_token_nodes, selected_indices)
            un_selected_indices = set(range(len(current_token_nodes))) - selected_indices
            # try to swap out each index
            found_better = False
            for i in selected_indices:
                for j in un_selected_indices:
                    new_selected_indices = selected_indices.copy()
                    new_selected_indices.remove(i)
                    new_selected_indices.add(j)
                    new_score = _calculate_activated_experts(current_token_nodes, new_selected_indices)
                    if new_score < best_score:
                        selected_indices = new_selected_indices
                        best_score = new_score
                        found_better = True
                        break
                if found_better:
                    break
            if not found_better:
                break
    results[layer_id] = (list(selected_indices), best_score)
    return results

def create_is_gt0_integer_var(model: gp.Model, x: gp.Var, big_M: float = 1000):
    """
    Create a variable that is 1 if x > 0, 0 otherwise.
    """
    z = model.addVar(vtype=gp.GRB.BINARY)
    model.addConstr(x <= big_M * z)
    return z

def find_best_tokens_greedy(current_token_nodes_dict: Dict[int, List[List[GraphNode]]],
                                  max_batch_size: int) -> List[int]:
    results = {}
    for layer_id, current_token_nodes in current_token_nodes_dict.items():
        effective_max_batch_size = min(max_batch_size, len(current_token_nodes))
        selected_indices = set()
        un_selected_indices = set(range(len(current_token_nodes)))
        selected_experts = set()
        def _update_selected_experts(new_indices):
            tmp_experts = set()
            for i in new_indices:
                for node in current_token_nodes[i]:
                    if isinstance(node, ExpertNode):
                        tmp_experts.add(node.expert_id)
            return selected_experts | tmp_experts
        while len(selected_indices) < effective_max_batch_size:
            min_score = float("inf")
            best_selected_indices = selected_indices.copy()
            best_selected_experts = selected_experts.copy()
            best_index = None
            for i in un_selected_indices:
                new_experts = _update_selected_experts([i])
                if len(new_experts) < min_score:
                    best_selected_indices = selected_indices | {i}
                    best_selected_experts = new_experts
                    best_index = i
                    min_score = len(new_experts)
            un_selected_indices.remove(best_index)
            selected_indices = best_selected_indices
            selected_experts = best_selected_experts
        results[layer_id] = (list(selected_indices), len(selected_experts))
    return results

def find_best_tokens(current_token_nodes_dict: Dict[int, List[List[GraphNode]]],
                   max_batch_size: int) -> List[int]:
    # create optimization variables
    # x_i = 1 if request i should be scheduled in the batch
    results = {}
    for layer_id, current_token_nodes in current_token_nodes_dict.items():
        if max_batch_size > len(current_token_nodes):
            max_batch_size = len(current_token_nodes)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            # env.setParam("MIPGap", 0.2)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addMVar(len(current_token_nodes), vtype=gp.GRB.BINARY)
                layer_expert_variables = {}
                layer_expert_seq_ids = {}
                for i, nodes_for_req_i in enumerate(current_token_nodes):
                    for node in nodes_for_req_i:
                        if isinstance(node, ExpertNode):
                            key = (node.layer_id, node.expert_id)
                            if key not in layer_expert_variables:
                                layer_expert_variables[key] = []
                            layer_expert_variables[key].append(x[i])
                            if key not in layer_expert_seq_ids:
                                layer_expert_seq_ids[key] = []
                            layer_expert_seq_ids[key].append(i)
                expert_activation_indicators = {}
                for key, variables in layer_expert_variables.items():
                    expert_activation_indicators[key]= create_is_gt0_integer_var(model, gp.quicksum(variables), 2 * len(variables))
                # objective: minimize number of experts activated
                model.setObjective(gp.quicksum(expert_activation_indicators.values()), gp.GRB.MINIMIZE)
                # constraint: sum of x_i <= max_batch_size
                model.addConstr(gp.quicksum(x) == max_batch_size)
                model.addConstr(x >= 0)
                model.addConstr(x <= 1)
                model._cur_obj = float('inf')
                model._time = time.time()
                model.optimize(callback=_early_stop_cb)
                # get top max_batch_size indices
                xis = [(i, x[i].X) for i in range(len(current_token_nodes))]
                xis = sorted(xis, key=lambda x: x[1], reverse=True)
                best_req_indices = [xi[0] for xi in xis[:max_batch_size]]
                activated_experts = _calculate_activated_experts(current_token_nodes, set(best_req_indices))
                results[layer_id] = (best_req_indices, activated_experts)
    return results

@register_schedule_strategy('PT')
class PrioritizeThroughput(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float,
                 n_experts_per_token: int,
                 graphs: List[RequestGraph],
                 ) -> None:
        self.n_layers: int = n_layers
        self.graphs: List[RequestGraph] = graphs
        self.n_experts_per_token: int = n_experts_per_token
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self._current_req_ids: List[int] = []
        self._current_phase: Optional[Tuple[int, type[GraphNode]]] = (0, AttnNode)
        self._activated_experts_history: List[int] = []

    def _advance_phase(self) -> None:
        if self._current_phase[1] == AttnNode:
            self._current_phase = (self._current_phase[0], ExpertNode)
        else:
            self._current_phase = (self._current_phase[0] + 1, AttnNode)
        if self._current_phase[0] == self.n_layers:
            # done with all layers
            self._current_phase = (0, AttnNode)
            self._current_req_ids = []

    def schedule(self,
                 stats_dict: Dict[int, RequestStats],
                 ready_nodes: List[GraphNode],
                 max_batch_size: Optional[int] = None,
                 ) -> Tuple[ModelComponent, List[GraphNode]]:
        if not ready_nodes:
            # no more requests to schedule
            return None, None
        if not self._current_req_ids:
            # get the first token of each request
            current_token_nodes: List[List[GraphNode]] = []
            request_init_node_ids: List[int] = []
            for node in ready_nodes:
                assert isinstance(node, AttnNode) and node.layer_id == 0
                assert node.layer_id == 0
                request_init_node_ids.append((node.req_id, node.node_id))
            request_init_node_ids = sorted(request_init_node_ids, key=lambda x: x[0])
            n_nodes_per_token = self.n_layers * (self.n_experts_per_token + 1)
            for req_id, node_id in request_init_node_ids:
                current_token_nodes.append(self.graphs[req_id].nodes[node_id:node_id + n_nodes_per_token])
            # find the best tokens
            t = time.time()
            best_token_indices, activated_experts = find_best_tokens_greedy({0: current_token_nodes}, max_batch_size)[0]
            self._activated_experts_history.append(activated_experts)
            # logger.info("Finding best tokens took {} seconds".format(time.time() - t))
            best_req_ids = sorted(set([current_token_nodes[i][0].req_id for i in best_token_indices]))
            current_token_nodes = [node for node in ready_nodes if node.req_id in best_req_ids]
            self._current_req_ids = best_req_ids
        filtered_ready_nodes = []
        for node in ready_nodes:
            if node.req_id in self._current_req_ids and \
                (node.layer_id, type(node)) == self._current_phase:
                filtered_ready_nodes.append(node)
        if not filtered_ready_nodes:
            # advance phase
            self._advance_phase()
            return self.schedule(stats_dict, ready_nodes, max_batch_size)
        if self._current_phase[1] == ExpertNode:
            # schedule one expert at a time
            filtered_ready_nodes = [node for node in filtered_ready_nodes if node.expert_id == filtered_ready_nodes[0].expert_id]
            return ModelComponent.EXPERT, filtered_ready_nodes
        else:
            # schedule all attn nodes together
            return ModelComponent.ATTENTION_GATE, filtered_ready_nodes


@register_schedule_strategy('PTL')
class PrioritizeThroughputLayerwise(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float,
                 n_experts_per_token: int,
                 graphs: List[RequestGraph],
                 ) -> None:
        self.n_layers: int = n_layers
        self.graphs: List[RequestGraph] = graphs
        self.n_experts_per_token: int = n_experts_per_token
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self._current_req_ids: List[int] = []
        self._current_layer_id: int = 0
        self._current_phase: type[GraphNode] = AttnNode
        self._activated_experts_history: List[int] = []

    def _advance_phase(self) -> None:
        if self._current_phase == AttnNode:
            self._current_phase = ExpertNode
        else:
            # done with this layer
            self._current_phase = AttnNode
            self._current_req_ids = []

    def schedule(self,
                 stats_dict: Dict[int, RequestStats],
                 ready_nodes: List[GraphNode],
                 max_batch_size: Optional[int] = None,
                 ) -> Tuple[ModelComponent, List[GraphNode]]:
        if not ready_nodes:
            # no more requests to schedule
            return None, None
        if not self._current_req_ids:
            # get the first token of each request
            current_token_nodes: List[List[GraphNode]] = []
            request_init_node_ids: List[int] = []
            for node in ready_nodes:
                assert isinstance(node, AttnNode)
                request_init_node_ids.append((node.req_id, node.node_id))
            request_init_node_ids = sorted(request_init_node_ids, key=lambda x: x[0])
            n_nodes_per_layer = self.n_experts_per_token + 1
            for req_id, node_id in request_init_node_ids:
                current_token_nodes.append(self.graphs[req_id].nodes[node_id:node_id + n_nodes_per_layer])
            # find the best tokens
            t = time.time()
            current_token_nodes_by_layer = {}
            for i, nodes in enumerate(current_token_nodes):
                layer_id = 0
                for node in nodes:
                    if isinstance(node, AttnNode):
                        layer_id = node.layer_id
                        break
                if layer_id not in current_token_nodes_by_layer:
                    current_token_nodes_by_layer[layer_id] = []
                current_token_nodes_by_layer[layer_id].append(nodes)
            grouping_results = find_best_tokens_greedy(current_token_nodes_by_layer, max_batch_size)
            selected_layer = None
            best_expert_over_bs = float('inf')
            for layer_id, (best_token_indices, activated_experts) in grouping_results.items():
                # how do we choose layers?
                # choose min activated experts / batch size
                expert_over_bs = activated_experts / len(best_token_indices)
                if selected_layer is None or expert_over_bs < best_expert_over_bs:
                    selected_best_token_indices = best_token_indices
                    selected_activated_experts = activated_experts
                    selected_layer = layer_id
            logger.debug("Selected {} tokens in layer {}, activated {} experts".format(
                len(selected_best_token_indices), selected_layer, selected_activated_experts))
            self._activated_experts_history.append(selected_activated_experts)
            # logger.info("Finding best tokens took {} seconds".format(time.time() - t))
            best_req_ids = sorted(set([current_token_nodes_by_layer[selected_layer][i][0].req_id for i in selected_best_token_indices]))
            self._current_req_ids = best_req_ids
            self._current_layer_id = selected_layer
        filtered_ready_nodes = []
        for node in ready_nodes:
            if node.req_id in self._current_req_ids and \
                node.layer_id == self._current_layer_id and \
                type(node) == self._current_phase:
                filtered_ready_nodes.append(node)
        if not filtered_ready_nodes:
            # advance phase
            self._advance_phase()
            return self.schedule(stats_dict, ready_nodes, max_batch_size)
        if self._current_phase == ExpertNode:
            # schedule one expert at a time
            filtered_ready_nodes = [node for node in filtered_ready_nodes if node.expert_id == filtered_ready_nodes[0].expert_id]
            return ModelComponent.EXPERT, filtered_ready_nodes
        else:
            # schedule all attn nodes together
            return ModelComponent.ATTENTION_GATE, filtered_ready_nodes


@register_schedule_strategy('PTLW')
class PrioritizeThroughputLayerwiseWithWait(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float,
                 n_experts_per_token: int,
                 graphs: List[RequestGraph],
                 min_candidates_per_expert: int = 1,
                 ) -> None:
        self.n_layers: int = n_layers
        self.graphs: List[RequestGraph] = graphs
        self.n_experts_per_token: int = n_experts_per_token
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self.min_candidates_per_expert: int = min_candidates_per_expert
        self._current_req_ids: List[int] = []
        self._current_layer_id: int = 0
        self._current_phase: type[GraphNode] = AttnNode
        self._activated_experts_history: List[int] = []

    def _advance_phase(self) -> None:
        if self._current_phase == AttnNode:
            self._current_phase = ExpertNode
        else:
            # done with this layer
            self._current_phase = AttnNode
            self._current_req_ids = []

    def schedule(self,
                 stats_dict: Dict[int, RequestStats],
                 ready_nodes: List[GraphNode],
                 max_batch_size: Optional[int] = None,
                 ) -> Tuple[ModelComponent, List[GraphNode]]:
        if not ready_nodes:
            # no more requests to schedule
            return None, None
        if not self._current_req_ids:
            # get the first token of each request
            current_token_nodes: List[List[GraphNode]] = []
            request_init_node_ids: List[int] = []
            for node in ready_nodes:
                assert isinstance(node, AttnNode)
                request_init_node_ids.append((node.req_id, node.node_id))
            request_init_node_ids = sorted(request_init_node_ids, key=lambda x: x[0])
            n_nodes_per_layer = self.n_experts_per_token + 1
            for req_id, node_id in request_init_node_ids:
                current_token_nodes.append(self.graphs[req_id].nodes[node_id:node_id + n_nodes_per_layer])
            # find the best tokens
            t = time.time()
            current_token_nodes_by_layer = {}
            for i, nodes in enumerate(current_token_nodes):
                layer_id = 0
                for node in nodes:
                    if isinstance(node, AttnNode):
                        layer_id = node.layer_id
                        break
                if layer_id not in current_token_nodes_by_layer:
                    current_token_nodes_by_layer[layer_id] = []
                current_token_nodes_by_layer[layer_id].append(nodes)
            s = "Avail. nodes per layer: "
            for layer_id in sorted(current_token_nodes_by_layer.keys()):
                nodes = current_token_nodes_by_layer[layer_id]
                s += "L{}: {}; ".format(layer_id, len(nodes))
            logger.debug(s)
            # filter out layers with too few candidates, except for layer 0
            layers_to_remove = []
            for layer_id, nodes in current_token_nodes_by_layer.items():
                if layer_id == 0:
                    continue
                if len(nodes) < self.min_candidates_per_expert:
                    layers_to_remove.append(layer_id)
            if len(layers_to_remove) < len(current_token_nodes_by_layer):
                for layer_id in layers_to_remove:
                    del current_token_nodes_by_layer[layer_id]
            logger.debug("Filtered layers: {}".format(layers_to_remove))
            grouping_results = find_best_tokens_greedy(current_token_nodes_by_layer, max_batch_size)
            selected_layer = None
            best_expert_over_bs = float('inf')
            for layer_id, (best_token_indices, activated_experts) in grouping_results.items():
                # how do we choose layers?
                # choose min activated experts / batch size
                expert_over_bs = activated_experts / len(best_token_indices)
                if selected_layer is None or expert_over_bs < best_expert_over_bs:
                    selected_best_token_indices = best_token_indices
                    selected_activated_experts = activated_experts
                    selected_layer = layer_id
            logger.debug("Selected {} tokens in layer {}, activated {} experts".format(
                len(selected_best_token_indices), selected_layer, selected_activated_experts))
            self._activated_experts_history.append(selected_activated_experts)
            best_req_ids = sorted(set([current_token_nodes_by_layer[selected_layer][i][0].req_id for i in selected_best_token_indices]))
            self._current_req_ids = best_req_ids
            self._current_layer_id = selected_layer
        filtered_ready_nodes = []
        for node in ready_nodes:
            if node.req_id in self._current_req_ids and \
                node.layer_id == self._current_layer_id and \
                type(node) == self._current_phase:
                filtered_ready_nodes.append(node)
        if not filtered_ready_nodes:
            # advance phase
            self._advance_phase()
            return self.schedule(stats_dict, ready_nodes, max_batch_size)
        if self._current_phase == ExpertNode:
            # schedule one expert at a time
            filtered_ready_nodes = [node for node in filtered_ready_nodes if node.expert_id == filtered_ready_nodes[0].expert_id]
            return ModelComponent.EXPERT, filtered_ready_nodes
        else:
            # schedule all attn nodes together
            return ModelComponent.ATTENTION_GATE, filtered_ready_nodes

