import os
from typing import List, Dict, Optional, Callable
import argparse
import tqdm


from vllm.transformers_utils.cost_model import ProfileBasedCostModel

from .dependency_graph import (
    build_graph_from_dataset,
    RequestGraph,
    GraphNode,
    DecodeNode,
)
from .schedule_strategies import (ScheduleStrategy,
                                  get_schedule_strategy,
                                  RequestStats)

from .logger import logger


class Simulator:
    """
    LLM serving simulator. Currently assumes all requests are known in advance.
    """
    def __init__(self, 
                 request_graphs: List[RequestGraph],
                 cost_model: ProfileBasedCostModel,
                 scheduler: ScheduleStrategy,
                 per_token_latency_slo_ms: float,
                 max_batch_size: int = 4096,
                 ) -> None:
        self.request_graphs: List[RequestGraph] = request_graphs
        self.cost_model: ProfileBasedCostModel = cost_model
        self.scheduler: ScheduleStrategy = scheduler
        self.per_token_latency_slo_ms: float = per_token_latency_slo_ms
        self.max_batch_size: int = max_batch_size
        self._latency_stats: Dict[int, RequestStats] = {}

    def get_ready_nodes(self,
                        filter: Optional[Callable[[GraphNode], bool]] = None
                        ) -> List[GraphNode]:
        ready_nodes: List[GraphNode] = []
        for graph in self.request_graphs:
            ready_nodes.extend(graph.get_frontier(filter))
        return ready_nodes

    def simulate(self):
        # follow some strategy to
        # 1. find which part of the model & which request to execute
        # 2. obtain the cost using the cost model
        # 3. update the latency/throughput stats for each request
        # 4. repeat until all requests are completed
        current_time = 0.0
        total_tokens_processed = 0
        # init _latency_stats
        for graph in self.request_graphs:
            self._latency_stats[graph.req_id] = RequestStats(graph.req_id,
                                                             enqueue_time=current_time)
        with tqdm.tqdm(total=len(self.request_graphs), desc="Running simulation") as pbar:
            while True:
                component, nodes_to_schedule = self.scheduler.schedule(
                    self._latency_stats,
                    self.get_ready_nodes(),
                    max_batch_size=self.max_batch_size
                )
                if not nodes_to_schedule:
                    logger.debug("No more nodes to schedule, exiting.")
                    break
                nodes_to_schedule: List[DecodeNode]
                # cost model needs both batch size and total tokens
                total_context_len = sum([
                    node.prompt_len + node.token_index for node in nodes_to_schedule
                ])
                batch_size = len(nodes_to_schedule)
                logger.debug("Got {} nodes to schedule, total context len: {}".format(
                    len(nodes_to_schedule), total_context_len
                ))
                cost = self.cost_model.get_cost(
                    component,
                    batch_size,
                    total_context_len
                )
                logger.debug("Cost: {} ms".format(cost))
                current_time += cost
                for node in nodes_to_schedule:
                    graph = self.request_graphs[node.req_id]
                    graph.execute(node)
                    if isinstance(node, DecodeNode) and node.is_last_node_for_token():
                        self._latency_stats[node.req_id].record_token_finish(current_time)
                        if len(self._latency_stats[node.req_id].get_per_token_latencies()) == len(self.request_graphs[node.req_id].decoded_token_ids) - 1:
                            pbar.update(1)
                        total_tokens_processed += 1
        self._throughput = total_tokens_processed / current_time * 1000.0

    def get_throughput(self) -> float:
        return self._throughput

    def get_latency_stats(self) -> Dict[int, RequestStats]:
        return self._latency_stats

    def get_first_token_latencies(self) -> List[float]:
        return [stats.first_token_latency() for stats in self._latency_stats.values()]

    def get_avg_token_latency(self, include_first_token: bool = False) -> List[float]:
        per_token_latencies = []
        for stats in self._latency_stats.values():
            per_req_latencies = stats.get_per_token_latencies()
            if include_first_token:
                per_token_latencies.append(per_req_latencies)
            else:
                per_token_latencies.append(per_req_latencies[1:])
        return sum(per_token_latencies) / len(per_token_latencies)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset-dir", type=str, required=True)
    parser.add_argument("-c", "--cost-model-dir", type=str, required=True)
    parser.add_argument("-s", "--strategy", type=str, required=True)
    parser.add_argument("-n", "--n-samples", type=int, default=1000)
    parser.add_argument("--max-batch-size", type=int, default=4096)
    parser.add_argument("--per-token-latency-slo-ms", type=float, default=1000.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args

def main(args):
    if args.debug:
        logger.setLevel("DEBUG")
    graphs, n_layers = build_graph_from_dataset(args.dataset_dir, max_samples=args.n_samples)
    print("Loaded {} graphs.".format(len(graphs)))

    cm_save_path = os.path.join(args.cost_model_dir, "cost_model.pkl")
    if os.path.exists(cm_save_path):
        cost_model = ProfileBasedCostModel.load(cm_save_path)
    else:
        cost_model = ProfileBasedCostModel(args.cost_model_dir)
        cost_model.save(cm_save_path)
    strategy = get_schedule_strategy(args.strategy,
                                     n_layers=n_layers,
                                     per_token_latency_slo_ms=args.per_token_latency_slo_ms)
    simulator = Simulator(graphs, cost_model, strategy, args.per_token_latency_slo_ms, args.max_batch_size)
    simulator.simulate()
    stats = simulator.get_latency_stats()
    print("Finished {} requests.".format(len(stats)))
    flattened_token_latencies = [latency for stat in stats.values() for latency in stat.get_per_token_latencies()[1:]]
    print("Avg latency: {} ms.".format(sum(flattened_token_latencies) / len(flattened_token_latencies)))
    print("Avg throughput: {} tokens/s.".format(simulator.get_throughput()))

if __name__ == "__main__":
    main(parse_args())