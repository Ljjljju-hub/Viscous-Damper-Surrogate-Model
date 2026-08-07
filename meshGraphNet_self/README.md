# Viscous Damper MeshGraphNet

This directory contains the complete training pipeline for the COMSOL HDF5 data:

1. `dataset.py` reads static topology and physical fields.
2. `mesh_motion.py` restores positions and ALE mesh velocities at time `t`.
3. `graph.py` converts triangular faces into dynamic edge features.
4. `model/` contains the reusable encode-process-decode MeshGraphNet.
5. `train.py` trains, validates, resumes, logs, and saves checkpoints.

Use the conda `pinn` environment for all subsequent commands:

```powershell
conda activate pinn
```

Install dependencies if that environment is rebuilt:

```powershell
python -m pip install -r meshGraphNet_self/requirements.txt
```

Run training from the repository root:

```powershell
python meshGraphNet_self/train.py --epochs 100 --batch-size 4
```

Checkpoint files are written under `meshGraphNet_self/checkpoints`:

- `last.pt`: latest complete training state
- `best.pt`: best validation model
- `epoch_XXXX.pt`: periodic snapshots
- `training_config.json`: command and model configuration

Resume training:

```powershell
python meshGraphNet_self/train.py --resume meshGraphNet_self/checkpoints/last.pt
```

Detailed Chinese design and operation documentation:

- [技术文档.md](技术文档.md)
