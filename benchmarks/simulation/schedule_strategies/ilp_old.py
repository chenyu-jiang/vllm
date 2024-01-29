from typing import List

import tqdm

import gurobipy as gp

from .base import ScheduleStrategy, register_schedule_strategy
from ..dependency_graph import AttnNode, ExpertNode
from ..logger import logger
from ..dependency_graph import RequestGraph

def create_is_gt0_integer_var(model: gp.Model, x: gp.Var, big_M: float = 1000):
    """
    Create a variable that is 1 if x > 0, 0 otherwise.
    """
    z = model.addVar(vtype=gp.GRB.BINARY)
    model.addConstr(x <= big_M * z)
    model.addConstr(x >= z)
    return z

def init_ilp_model(request_graphs: List[RequestGraph],
                   k_experts_per_token: int,
                   attn_cost: float,
                   expert_cost: float,
                   max_T: int = None):
    """
    Initialize the ILP model.
    """
    # print expert assignment for each request
    for i, graph in enumerate(request_graphs):
        expert_per_layer = {}
        for node in graph.nodes:
            if isinstance(node, ExpertNode):
                if node.layer_id not in expert_per_layer:
                    expert_per_layer[node.layer_id] = []
                expert_per_layer[node.layer_id].append(node.expert_id)
        print("Request {}: {}".format(i, expert_per_layer))
    # create optimization variables
    # x_{i,j,t} = 1 if request i's node j is processed at time step t
    model = gp.Model()
    logger.debug("Initializing ILP model...")
    max_js = [len(graph.nodes) - 1 for graph in request_graphs] # last node is FinNode
    logger.debug("Max js: {}".format(max_js))
    if not max_T:
        # worst case: all nodes executed separately
        max_T = sum(max_js)
    x_per_request = []
    for i in tqdm.trange(len(request_graphs), desc="Creating variables Xs"):
        xs = model.addMVar((max_js[i], max_T), vtype=gp.GRB.BINARY)
        x_per_request.append(xs)
    logger.debug("Created variable Xs.")

    # Objective: minimize makespan (max throughput)
    attn_costs = [] # list, shape (t,)
    expert_costs = [] # list, shape (t,)
    logger.debug("Creating aux variable for the objective...")
    for t in tqdm.trange(max_T, desc="Aux variables for objective"):
        attn_nodes_vars = []
        expert_nodes_vars = []
        for i in range(len(request_graphs)):
            for j in range(max_js[i]):
                if j % (k_experts_per_token + 1) == 0:
                    attn_nodes_vars.append(x_per_request[i][j, t])
                else:
                    expert_nodes_vars.append(x_per_request[i][j, t])
        aux_z_attn = create_is_gt0_integer_var(model, gp.quicksum(attn_nodes_vars))
        aux_z_expert = create_is_gt0_integer_var(model, gp.quicksum(expert_nodes_vars))
        attn_costs.append(aux_z_attn * attn_cost)
        expert_costs.append(aux_z_expert * expert_cost)
    model.setObjective(gp.quicksum(attn_costs) + gp.quicksum(expert_costs), gp.GRB.MINIMIZE)
    logger.debug("Created objective.")
    # Constraints
    # 1. every node is processed exactly once
    logger.debug("Creating constraint 1...")
    node_exec_once_constraints = []
    for i in range(len(request_graphs)):
        for j in range(max_js[i]):
            node_exec_once_constraints.append(gp.quicksum(x_per_request[i][j, :]) == 1)
    model.addConstrs(c for c in node_exec_once_constraints)
    logger.debug("Created constraint 1.")
    # 2. only one type of node is processed at each time step
    #    here type = attn/expert + layer_id + expert_id (if is expert)
    # we first organize nodes by type
    logger.debug("Creating constraint 2...")
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
            nodes_by_type[node_type].append((i, j))
    # then create constraints
    for t in tqdm.trange(max_T, desc="Creating node type constraints"):
        per_node_type_vars = []
        for nodes in (nodes_by_type.values()):
            node_type_vars = []
            for i, j in nodes:
                node_type_vars.append(x_per_request[i][j, t])
            is_this_type_active_at_time_t = create_is_gt0_integer_var(model, gp.quicksum(node_type_vars))
            per_node_type_vars.append(is_this_type_active_at_time_t)
        model.addConstr(gp.quicksum(per_node_type_vars) <= 1)
    logger.debug("Created constraint 2.")
    # 3. dependency constraints
    logger.debug("Creating constraint 3...")
    dependency_constraints = []
    # first create aux variables indicating whether node j has been executed
    # after time step t
    node_executed_vars = []
    for i in tqdm.trange(len(request_graphs), desc="Creating aux variables for constraint 3"):
        j_list = []
        for j in range(max_js[i]):
            t_list = [] 
            for t in range(1, max_T + 1):
                aux_z = create_is_gt0_integer_var(model, gp.quicksum(x_per_request[i][j, :t]))
                t_list.append(aux_z)
            j_list.append(t_list)
        node_executed_vars.append(j_list)
    # then create dependency constraints
    for i in tqdm.trange(len(request_graphs), desc="Creating dependency constraints"):
        for j in range(max_js[i]):
            node = request_graphs[i].nodes[j]
            for parent_node in node.parents:
                parent_node_idx = request_graphs[i].nodes.index(parent_node)
                for t in range(1, max_T):
                    dependency_constraints.append(node_executed_vars[i][j][t] <= node_executed_vars[i][parent_node_idx][t-1])
    model.addConstrs(c for c in dependency_constraints)
    logger.debug("Created constraint 3.")
    # 4. memory constraints
    # ignore for now for testing
    logger.debug("Created problem. Calling solver...")
    # save model for inspection
    model.write('ilp.mps')
    model.optimize()
    def _print_nodes_run_at_t(t):
        # group nodes by type
        nodes_by_type = {}
        for i in range(len(x_per_request)):
            for j in range(x_per_request[i].shape[0]):
                if x_per_request[i][j, t].X > 0:
                    node = request_graphs[i].nodes[j]
                    if isinstance(node, AttnNode):
                        key = (type(node), node.layer_id, 0)
                    elif isinstance(node, ExpertNode):
                        key = (type(node), node.layer_id, node.expert_id)
                    else:
                        # FinNode, ignore
                        continue
                    if key not in nodes_by_type:
                        nodes_by_type[key] = []
                    nodes_by_type[key].append((i, j))
        # print
        for key, nodes in nodes_by_type.items():
            if key[0] == AttnNode:
                print("AttnNode, layer {}, t={}, with request ids: {}".format(key[1], t, [n[0] for n in nodes]))
            elif key[0] == ExpertNode:
                print("ExpertNode, layer {}, expert_id {}, t={}, with request ids: {}".format(key[1], key[2], t, [n[0] for n in nodes]))
    import code
    code.interact(local=locals())

@register_schedule_strategy('ILP')
class ILP(ScheduleStrategy):
    def __init__(self,
                 n_layers: int,
                 per_token_latency_slo_ms: float,
                 request_graphs: int,
                 k_experts_per_token: int,
                 max_T: int = None
                 ) -> None:
        self.request_graphs = request_graphs
        self.n_layers: int = n_layers
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self.k_experts_per_token: int = k_experts_per_token
        init_ilp_model(request_graphs, k_experts_per_token, 1, 1, max_T=max_T)