# Reproducibility

This repository contains code and compact result artifacts, but not the external datasets or model checkpoints.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build that matches your machine or cluster. The thesis experiments used Python 3.10.19, PyTorch 2.7.1+cu118, Transformers 4.45.2, and an NVIDIA H200 GPU.

## Data Preparation

Place Flickr30K and SciCap files under the paths configured in [`../configs/experiment_config.yaml`](../configs/experiment_config.yaml). For SciCap:

```bash
python -m src.prepare_scicap_split --snapshot data/scicap_source
```

The repository does not redistribute Flickr30K, SciCap, or raw dataset images.

## Evaluation

```bash
python -m src.eval.evaluate_clip --config configs/experiment_config.yaml --dataset flickr30k --split test
python -m src.eval.evaluate_blip2 --config configs/experiment_config.yaml --dataset scicap --split test --rerank-top-k 128
```

Full BLIP-2 evaluation and fine-tuning should be run on a GPU node. Sanitized SLURM launchers are provided in [`../scripts/slurm`](../scripts/slurm). Set `PROJECT_ROOT`, `CONDA_INIT`, and `CONDA_ENV` if your cluster layout differs.

## Figures

```bash
python -m src.analysis.plot_training_curves
```

Regenerating qualitative SciCap collages requires fine-tuned checkpoints. Set `CLIP_SCICAP_CKPT` and `BLIP2_SCICAP_CKPT` to local checkpoint paths before running `python -m src.analysis.make_collages`.
