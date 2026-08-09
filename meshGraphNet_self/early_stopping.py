import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStoppingUpdate:
    best_valid_loss: float
    reference_loss: float
    epochs_without_improvement: int
    exact_improvement: bool
    meaningful_improvement: bool
    relative_improvement: float


def update_early_stopping(
    valid_loss: float,
    best_valid_loss: float,
    reference_loss: float,
    epochs_without_improvement: int,
    min_relative_improvement: float,
) -> EarlyStoppingUpdate:
    if not 0.0 <= min_relative_improvement < 1.0:
        raise ValueError("min_relative_improvement must be in [0, 1).")

    exact_improvement = valid_loss < best_valid_loss
    new_best_valid_loss = min(best_valid_loss, valid_loss)

    if not math.isfinite(reference_loss):
        relative_improvement = 1.0
        meaningful_improvement = True
    elif reference_loss == 0.0:
        relative_improvement = 0.0
        meaningful_improvement = False
    else:
        relative_improvement = (reference_loss - valid_loss) / abs(reference_loss)
        if min_relative_improvement == 0.0:
            meaningful_improvement = valid_loss < reference_loss
        else:
            meaningful_improvement = valid_loss <= reference_loss * (
                1.0 - min_relative_improvement
            )

    return EarlyStoppingUpdate(
        best_valid_loss=new_best_valid_loss,
        reference_loss=(valid_loss if meaningful_improvement else reference_loss),
        epochs_without_improvement=(
            0 if meaningful_improvement else epochs_without_improvement + 1
        ),
        exact_improvement=exact_improvement,
        meaningful_improvement=meaningful_improvement,
        relative_improvement=relative_improvement,
    )


def reference_from_checkpoint(checkpoint: dict, best_valid_loss: float) -> float:
    return float(checkpoint.get("early_stopping_reference_loss", best_valid_loss))
