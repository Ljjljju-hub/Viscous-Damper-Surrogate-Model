"""Internal MeshGraphNet worker launched by the root training entry point."""

import argparse
from pathlib import Path

try:
    from .dataset import CASE_FEATURE_NAMES
    from .model.simulator import SurrogateSimulator
    from .training import FIELD_NAMES, add_common_training_args, run_training
except ImportError:
    from dataset import CASE_FEATURE_NAMES
    from model.simulator import SurrogateSimulator
    from training import FIELD_NAMES, add_common_training_args, run_training


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    self_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Internal MeshGraphNet worker. Use training_workspace/train.py."
    )
    add_common_training_args(parser, project_root, self_root)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--message-passing-steps", type=int, default=15)
    return parser.parse_args()


def main(args=None):
    args = parse_args() if args is None else args
    model_config = {
        "field_count": len(FIELD_NAMES),
        "case_feature_count": len(CASE_FEATURE_NAMES),
        "region_count": 3,
        "edge_input_size": 3,
        "hidden_size": args.hidden_size,
        "message_passing_steps": args.message_passing_steps,
    }
    return run_training(
        args,
        model_config=model_config,
        model_factory=lambda: SurrogateSimulator(**model_config),
        model_name="meshgraphnet",
    )


if __name__ == "__main__":
    main()
