import argparse

from dependency_graph import (
    build_graph_from_dataset,
    RequestGraph,
    AttnNode,
    ExpertNode,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    graphs = build_graph_from_dataset(args.dataset_dir)
    import code
    code.interact(local=locals())


if __name__ == "__main__":
    main(parse_args())