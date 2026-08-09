# Relative Early-Stopping Design

## Goal

Prevent insignificant validation-loss fluctuations from resetting early stopping, while still saving the checkpoint with the exact lowest validation loss.

## Configuration

The shared training entry point exposes:

```python
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_RELATIVE_IMPROVEMENT = 0.002  # 0.2%
```

Both values are forwarded to MeshGraphNet and Transolver workers and recorded in the run request and training configuration.

## Two Independent Best Values

Training tracks two values:

1. `best_valid_loss`: exact lowest validation loss. Any strict decrease updates `best.pt`.
2. `early_stopping_reference_loss`: last validation loss that improved the previous reference by at least the configured relative threshold.

For reference loss `r` and current loss `v`, an improvement is meaningful when:

```text
v < r * (1 - min_relative_improvement)
```

The first completed validation epoch always initializes both values.

## Epoch Behavior

After validation:

1. Step `ReduceLROnPlateau` with the raw validation loss.
2. Save `best.pt` whenever the raw validation loss is an exact new minimum.
3. Reset the early-stop counter only for a meaningful relative improvement.
4. Otherwise increment the counter.
5. Stop after the counter reaches `patience`.

The epoch that reaches patience is fully evaluated and may still become the exact `best.pt` before training stops.

## Logging and Summary

Each epoch line reports:

```text
early_stop_wait=<current>/<patience>
relative_improvement=<percentage>
meaningful_improvement=<True|False>
```

The checkpoint stores the reference loss and counter. The final summary records the threshold, patience, final counter, and reference loss.

## Resume Compatibility

New checkpoints restore both exact-best and early-stop-reference values. A legacy checkpoint without the reference field uses its exact `best_valid_loss` as the initial reference and preserves its existing counter.

Changing the source does not alter a running Python process. The current MeshGraphNet worker must be stopped and restarted from `last.pt` before the new rule applies.

## Tests

Automated tests cover:

1. A tiny exact improvement saves the best value but does not reset patience.
2. A 0.2% or greater improvement resets patience and updates the reference.
3. Patience 10 stops on the expected epoch.
4. Legacy checkpoint state receives a compatible reference value.
5. Replaying the completed Transolver validation curve stops at epoch 88, retains epoch 88 as the exact best checkpoint, and keeps epoch 78 as the meaningful-improvement reference.

The learning-rate scheduler and model architecture are unchanged.
