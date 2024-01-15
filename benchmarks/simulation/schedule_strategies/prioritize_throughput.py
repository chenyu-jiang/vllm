from typing import Dict, List, Optional, Tuple
from .base import ScheduleStrategy, register_schedule_strategy
from ..simulator import RequestStats
from ..dependency_graph import GraphNode, AttnNode, ExpertNode
from ..logger import logger
from vllm.transformers_utils.cost_model import ModelComponent

@register_schedule_strategy('PT')
class PrioritizeThroughput(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float
                 ) -> None:
        self.n_layers: int = n_layers
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms

    def schedule(self,
                 stats_dict: Dict[int, RequestStats],
                 ready_nodes: List[GraphNode],
                 max_batch_size: Optional[int] = None,
                 ) -> Tuple[ModelComponent, List[GraphNode]]:
        if not ready_nodes:
            # no more requests to schedule
            return None, None
        # group ready nodes by phase (i.e., layer + attn/expert + expert_id)
        ready_nodes_by_phase: Dict[Tuple[int, type[GraphNode]], List[GraphNode]] = {}
        for node in ready_nodes:
            if isinstance(node, AttnNode):
                phase = (node.layer_id, type(node), 0)
            elif isinstance(node, ExpertNode):
                phase = (node.layer_id, type(node), node.expert_id)
            if phase not in ready_nodes_by_phase:
                ready_nodes_by_phase[phase] = []
            ready_nodes_by_phase[phase].append(node)
        # select the phase with the most available nodes
        (selected_layer, _, selected_expert_id), current_batch_nodes = max(ready_nodes_by_phase, key=lambda x: len(x[1]))
        if self._current_phase == ExpertNode:
            # schedule one expert at a time
            logger.debug("Scheduling {} expert nodes: Layer {}, Expert {}".format(
                len(current_batch_nodes), selected_layer, selected_expert_id)
            )
            return ModelComponent.EXPERT, current_batch_nodes
        else:
            # schedule all attn nodes together
            logger.debug("Scheduling {} attn nodes: Layer {}".format(len(current_batch_nodes), selected_layer))
            return ModelComponent.ATTENTION_GATE, current_batch_nodes

