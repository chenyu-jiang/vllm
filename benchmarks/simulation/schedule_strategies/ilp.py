from typing import List

import tqdm

import gurobipy as gp

from .base import ScheduleStrategy, register_schedule_strategy
from ..dependency_graph import AttnNode, ExpertNode
from ..logger import logger
from ..dependency_graph import RequestGraph

def one_if_positive(model: gp.Model, x: gp.Var, big_M: float = 1000):
    """
    Create a variable that is 1 if x > 0
    """
    z = model.addVar(vtype=gp.GRB.BINARY)
    model.addConstr(x <= big_M * z)
    return z

def add_disjunctive_constraints(model: gp.Model, exprs: List[gp.LinExpr], big_M: float = 1000):
    zs = model.addMVar(len(exprs), vtype=gp.GRB.BINARY)
    for i, expr in enumerate(exprs):
        model.addConstr(expr <= big_M * (1 - zs[i]))
    model.addConstr(gp.quicksum(zs) >= 1)


def solve_using_ilp_model(request_graphs: List[RequestGraph],
                   n_experts_per_token: int,
                   attn_cost: float,
                   expert_cost: float,
                   max_batch_size: int = None,
                  ) -> List[int]:
    """
    Construct and solve the ILP model.
    """
    # create optimization variables
    # t_{i,j}: start time of request i's node j
    model = gp.Model()
    logger.debug("Initializing ILP model...")
    max_js = [len(graph.nodes) - 1 for graph in request_graphs] # last node is FinNode
    logger.debug("Max js: {}".format(max_js))
    t_per_request = []
    for i in tqdm.trange(len(request_graphs), desc="Creating variables ts"):
        ts = model.addMVar(max_js[i], lb=0, vtype=gp.GRB.CONTINUOUS)
        t_per_request.append(ts)
    logger.debug("Created variable ts.")

    # Objective: minimize makespan (max throughput)
    t_max = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    model.setObjective(t_max, gp.GRB.MINIMIZE)
    # relate t_max to t_{i,j}
    for i in tqdm.trange(len(request_graphs), desc="Creating constraints for objective"):
        model.addConstr(t_max >= t_per_request[i])
        for j in range(max_js[i]):
            model.addConstr(t_per_request[i][j] <= t_max)
    logger.debug("Created objective.")
    # Constraints
    # 1. Nodes that do not belong to the same type cannot overlap
    #    here type = attn/expert + layer_id + expert_id (if is expert)
    # we first organize nodes by type
    logger.debug("Creating constraint 1...")
    nodes_by_type = {}
    for i, graph in tqdm.tqdm(enumerate(request_graphs), total=len(request_graphs), desc="Organizing nodes by type"):
        for j, node in enumerate(graph.nodes):
            if isinstance(node, AttnNode):
                node_type = (type(node), node.layer_id, 0)
            elif isinstance(node, ExpertNode):
                node_type = (type(node), node.layer_id, node.expert_id)
            else:
                # FinNode, ignore
                continue
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(node)
    # then create constraints
    unique_types = list(nodes_by_type.keys())
    for type_i in tqdm.trange(len(unique_types), desc="Creating constraints for constraint 1"):
        for type_j in range(type_i + 1, len(unique_types)):
            # type_i and type_j are different
            nodes_i = nodes_by_type[unique_types[type_i]]
            nodes_j = nodes_by_type[unique_types[type_j]]
            for node_i in nodes_i:
                for node_j in nodes_j:
                    req_id_i = node_i.req_id
                    req_id_j = node_j.req_id
                    if req_id_i == req_id_j:
                        # no need to add this constraint
                        # since they will be covered by dependency constraints
                        continue
                    time_node_i = attn_cost if isinstance(node_i, AttnNode) else expert_cost
                    time_node_j = attn_cost if isinstance(node_j, AttnNode) else expert_cost
                    add_disjunctive_constraints(model,
                                                [t_per_request[req_id_i][node_i.node_id] + time_node_i - t_per_request[req_id_j][node_j.node_id],
                                                 t_per_request[req_id_j][node_j.node_id] + time_node_j - t_per_request[req_id_i][node_i.node_id]])
    logger.debug("Created constraint 1.")
    # 2. dependency constraints
    logger.debug("Creating constraint 3...")
    for i in tqdm.trange(len(request_graphs), desc="Creating dependency constraints"):
        for j in range(max_js[i]):
            node = request_graphs[i].nodes[j]
            for parent_node in node.parents:
                parent_node_idx = request_graphs[i].nodes.index(parent_node)
                parent_node_exec_time = attn_cost if isinstance(parent_node, AttnNode) else expert_cost
                model.addConstr(t_per_request[i][j] >= t_per_request[i][parent_node_idx] + parent_node_exec_time)
    logger.debug("Created constraint 3.")
    # 4. memory constraints
    # ignore for now for testing
    logger.debug("Created problem. Calling solver...")
    # save model for inspection
    model.write('ilp.mps')
    model.optimize()
    import code
    code.interact(local=locals())

@register_schedule_strategy('ILP')
class ILP(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float,
                 request_graphs: int,
                 n_experts_per_token: int,
                 max_T: int = None
                 ) -> None:
        self.request_graphs = request_graphs
        self.n_layers: int = n_layers
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self.n_experts_per_token: int = n_experts_per_token
        solve_using_ilp_model(request_graphs, n_experts_per_token, 1, 1)