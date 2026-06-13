# Toward Large CGRA Mapping Models

## Paper Status
Under review.

## Summary

This repository contains the implementation associated with the paper Towards Large CGRA Mapping Models. It focuses on supervised learning-based CGRA mapping, with an emphasis on a new transformer-based model called the General Placement and Routing Model (GPRM), and a novel mapping generator called the Grammar-based Optimal Mapping Generator (GOMG, https://github.com/El3ci0Sz/GOMG-CGRA).


## Setup

Clone the repository and build the provided container image:

```bash
git clone https://github.com/lesc-ufv/GPRM.git
cd GPRM
docker build -t gprm .
```

Run the project inside the container with GPU access, mounting the cloned repository into `/workspace`, and keep it alive in detached mode:

```bash
docker run -d \
  --name gprm-dev \
  --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  gprm \
  tail -f /dev/null
```

Open a shell inside the container:

```bash
docker exec -it gprm-dev bash
```

Then build the binaries from `/workspace` with:

```bash
./rebuild.sh
```


## Reproducibility

The repository includes benchmark data under `benchmarks/` and experiment entry points under `experiments/mapping_representation_learning/`. The default workflow couples a Python server with C++ experiment binaries.

### Minimal experiment flow

1. Start the Python server in a `screen` session:

```bash
screen -S server /bin/bash
python3 server.py --port 8000 --num-workers 4
```

Detach from the session without stopping the server:

```bash
Ctrl+A, then D
```

2. In another terminal, launch an experiment binary from `build/`, also inside `screen` if you want the job to keep running after disconnecting:

```bash
screen -S train /bin/bash

./build/run_gprm_mrl_exp \
  benchmarks/16_supervised_bench_4x4 \
  benchmarks/test/ \
  8000 \
  32 \
  /cache/gprm_supervised_data \
  128 \
  32 \
  8 \
  8
```

Detach from the training session with:

```bash
Ctrl+A, then D
```

You can reattach later with:

```bash
screen -r train
```

3. Training outputs, checkpoints, and TensorBoard logs are written under `results/mapping_representation_learning/`.

### TensorBoard

Inside the container, start TensorBoard with:

```bash
tensorboard --logdir results/mapping_representation_learning --port 6006 --bind_all
```
### Experiment templates

The current `experiments/mapping_representation_learning/` directory provides five entry points. All of them share the same first five arguments:

```bash
<train_dot_dir> <eval_dot_dir> <port> <num_threads> <path_to_supervised_data>
```

#### `run_gprm_mrl_exp`

```bash
./build/run_gprm_mrl_exp \
  <train_dot_dir> \
  <eval_dot_dir> \
  <port> \
  <num_threads> \
  <path_to_supervised_data> \
  <out_dim> \
  <feature_out_dim> \
  <n_heads> \
  <n_transformer_layers>
```

#### `run_gprm_ablation`

```bash
./build/run_gprm_ablation \
  <train_dot_dir> \
  <eval_dot_dir> \
  <port> \
  <num_threads> \
  <path_to_supervised_data> \
  <component_to_remove> \
  <gin_layers>
```

Typical values for `<component_to_remove>` include `MLP`, `RMSNorm`, `NormBeforePred`, `GinEmb`, `Mobility`, `Coord`, `GNN`, `Transformer`, `PlacementOrder`, `IgnoreLPE`, `NormInitialEmb`, and `UseScale`.

#### `run_smartmap_mrl_exp`

```bash
./build/run_smartmap_mrl_exp \
  <train_dot_dir> \
  <eval_dot_dir> \
  <port> \
  <num_threads> \
  <path_to_supervised_data> \
  <out_dim> \
  <n_heads>
```

#### `run_mapzero_mrl_exp`

```bash
./build/run_mapzero_mrl_exp \
  <train_dot_dir> \
  <eval_dot_dir> \
  <port> \
  <num_threads> \
  <path_to_supervised_data> \
  <out_dim> \
  <n_heads>
```

#### `run_transmap_mrl_exp`

```bash
./build/run_transmap_mrl_exp \
  <train_dot_dir> \
  <eval_dot_dir> \
  <port> \
  <num_threads> \
  <path_to_supervised_data> \
  <out_dim> \
  <n_heads> \
  <num_mhd_layers> \
  <dropout>
```

These templates reflect the current source files in this repository snapshot. Concrete values for dataset paths, supervised blocks, and hyperparameters should match the configuration used in the manuscript run being reproduced.





## Online Appendix

It is worth pointing readers to the appendix. The main manuscript benefits from a short reference to the additional ablation material, especially because the appendix documents the contribution of individual GPRM components more directly than the main text.

The current appendix material is available at:

- `appendix/gprm_ablation.md`

This file reports the architecture ablation study and should be cited from the README as the natural place for supplementary experimental detail.

## Repository Structure

- `src/`: core C++ and Python implementation
- `experiments/`: experiment drivers for training and ablation studies
- `scripts/`: utility binaries for dataset generation, filtering, augmentation, and zero-shot mapping
- `cli/`: Python preprocessing and graph-cleaning tools
- `benchmarks/`: benchmark DFG collections used by the experiments
- `appendix/`: supplementary material referenced by the manuscript

# Citation
Waiting review.