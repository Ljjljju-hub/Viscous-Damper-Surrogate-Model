import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from meshGraphNet_self.dataset import CASE_FEATURE_NAMES
from meshGraphNet_self.training import (
    FIELD_NAMES,
    add_common_training_args,
    checkpoint_state,
    choose_device,
    create_dataloader,
    evaluate,
    restore_checkpoint,
    run_training,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)
from transolver_self.model.simulator import TransolverSimulator


OFFICIAL_TRANSOLVER_REVISION = "75e0f67643806a81cd1d3f6adc88dd8c02416fe7"


def parse_args():
    self_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train Transolver on the viscous-damper moving mesh."
    )
    add_common_training_args(parser, PROJECT_ROOT, self_root)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--slice-num", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-ratio", type=int, default=1)
    return parser.parse_args()


def build_model_config(args) -> dict:
    return {
        "field_count": len(FIELD_NAMES),
        "case_feature_count": len(CASE_FEATURE_NAMES),
        "region_count": 3,
        "layers": args.layers,
        "hidden_size": args.hidden_size,
        "heads": args.heads,
        "slice_num": args.slice_num,
        "dropout": args.dropout,
        "mlp_ratio": args.mlp_ratio,
    }


def main(args=None):
    args = parse_args() if args is None else args
    model_config = build_model_config(args)
    return run_training(
        args,
        model_config=model_config,
        model_factory=lambda: TransolverSimulator(**model_config),
        model_name="transolver",
        extra_config={
            "model": "THUML Transolver irregular mesh",
            "official_revision": OFFICIAL_TRANSOLVER_REVISION,
        },
        extra_checkpoint={
            "official_revision": OFFICIAL_TRANSOLVER_REVISION,
        },
    )


if __name__ == "__main__":
    main()
