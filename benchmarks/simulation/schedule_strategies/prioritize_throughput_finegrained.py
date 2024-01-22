from typing import Dict, List, Optional, Tuple
from .base import ScheduleStrategy, register_schedule_strategy
from ..simulator import RequestStats
from ..dependency_graph import GraphNode, AttnNode, ExpertNode
from ..logger import logger
from vllm.transformers_utils.cost_model import ModelComponent

@register_schedule_strategy('PTFG')
class PrioritizeThroughputFineGrained(ScheduleStrategy):
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
        has_attn_nodes = False
        for node in ready_nodes:
            if isinstance(node, AttnNode):
                has_attn_nodes = True
                phase = (node.layer_id, type(node), 0)
            elif isinstance(node, ExpertNode):
                phase = (node.layer_id, type(node), node.expert_id)
            if phase not in ready_nodes_by_phase:
                ready_nodes_by_phase[phase] = []
            ready_nodes_by_phase[phase].append(node)
        logger.debug("Ready nodes by phase: ")
        for phase, nodes in ready_nodes_by_phase.items():
            if phase[1] == AttnNode:
                logger.debug("\tLayer {} Attn : {} nodes".format(phase[0], len(nodes)))
            else:
                logger.debug("\tLayer {} Expert {}: {} nodes".format(phase[0], phase[2], len(nodes)))
        # select the phase with the most available nodes, prioritizing attn nodes
        if has_attn_nodes:
            (selected_layer, selected_node_type, selected_expert_id), current_batch_nodes = max([(k, v) for k, v in ready_nodes_by_phase.items() if k[1] == AttnNode], key=lambda x: len(x[1]))
        else:
            (selected_layer, selected_node_type, selected_expert_id), current_batch_nodes = max(ready_nodes_by_phase.items(), key=lambda x: len(x[1]))
        if selected_node_type == ExpertNode:
            # schedule one expert at a time
            logger.debug("Scheduling {} expert nodes: Layer {}, Expert {}".format(
                len(current_batch_nodes), selected_layer, selected_expert_id)
            )
            return ModelComponent.EXPERT, current_batch_nodes
        else:
            # schedule all attn nodes together
            logger.debug("Scheduling {} attn nodes: Layer {}".format(len(current_batch_nodes), selected_layer))
            return ModelComponent.ATTENTION_GATE, current_batch_nodes