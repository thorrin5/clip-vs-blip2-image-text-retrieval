# Comparative Analysis of CLIP vs BLIP-2 for Image-Text Retrieval

Bachelor thesis project by **Tibor Horcsin**  
Technical University of Kosice, Faculty of Electrical Engineering and Informatics  
Supervisor: **prof. Ing. Peter Sincak, CSc.**

This repository accompanies the practical part of the bachelor thesis **Comparative Analysis of CLIP vs BLIP-2 for Image-Text Retrieval**. It compares CLIP and BLIP-2 for bidirectional image-text retrieval, focusing on retrieval accuracy, computational cost, and transfer from natural-image captions to scientific figure-caption data.

The repository contains the active evaluation and fine-tuning source code, compact result artifacts, raw JSON metric payloads, training histories, plots, and a small set of qualitative retrieval collages. It does **not** include datasets, model checkpoints, model caches, downloaded papers, or the full thesis PDF.

## Models

| Model | Implementation used | Retrieval method |
|---|---|---|
| CLIP ViT-L/14 | Official OpenAI CLIP package, model `ViT-L/14` | Independent image/text embeddings with normalized dot-product similarity |
| BLIP-2 ViT-g ITM | Hugging Face Transformers, model `Salesforce/blip2-itm-vit-g` | ITC candidate scoring followed by ITM reranking of top candidates |

The source code uses the official OpenAI `clip` package for CLIP, not OpenCLIP. BLIP-2 is loaded through `transformers.Blip2ForImageTextRetrieval`.

## Datasets

Datasets are not included in this repository.

| Dataset | Use in this project | Notes |
|---|---|---|
| Flickr30K | Zero-shot retrieval benchmark | Karpathy test split, 1,000 images and 5,000 captions |
| SciCap | Scientific-domain transfer and fine-tuning | 45,000 train, 1,000 validation, 1,000 test pairs created with seed 42 |
| AI2D | Historical exploratory experiments | Retained in results for traceability, not the final thesis focus |

## Evaluation

The retrieval task is evaluated in both directions:

- image-to-text retrieval
- text-to-image retrieval
- Recall@1, Recall@5, Recall@10
- Mean Recall@K, computed as the average of image-to-text and text-to-image Recall@K
- runtime and throughput fields where measured

## Main Results

Values below are copied from [`results/full_comparison.csv`](results/full_comparison.csv). Runtime is total measured evaluation time in seconds.

| Dataset | Model | Setting | Mean R@1 | Mean R@5 | Mean R@10 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Flickr30K | CLIP | zero-shot | 75.06 | 92.34 | 95.56 | 5 |
| Flickr30K | BLIP-2 | zero-shot | 91.55 | 98.37 | 99.19 | 4247 |
| AI2D | CLIP | zero-shot | 8.55 | 26.77 | 37.10 | 3 |
| AI2D | BLIP-2 | zero-shot | 10.65 | 26.13 | 37.42 | 701 |
| AI2D | CLIP | fine-tuned | 16.94 | 39.35 | 49.84 | 3 |
| AI2D | BLIP-2 | fine-tuned | 15.97 | 39.68 | 52.90 | 1551 |
| SciCap | CLIP | zero-shot | 21.65 | 33.95 | 40.25 | 13 |
| SciCap | BLIP-2 | zero-shot | 26.30 | 35.55 | 39.95 | 1988 |
| SciCap | CLIP | fine-tuned | 33.95 | 50.55 | 58.10 | 10 |
| SciCap | BLIP-2 | fine-tuned | 30.00 | 42.20 | 47.75 | 5195 |

On Flickr30K, BLIP-2 achieved higher retrieval accuracy but was much slower because ITM reranking evaluates many candidate image-text pairs. On SciCap, BLIP-2 was stronger zero-shot, while CLIP adapted more effectively after fine-tuning under the reported setup.

## Repository Contents

| Path | Contents |
|---|---|
| [`src/models`](src/models) | CLIP and BLIP-2 wrappers |
| [`src/data`](src/data) | Dataset loaders for Flickr30K, SciCap, and AI2D metadata |
| [`src/eval`](src/eval) | Evaluation entrypoints and Recall@K metrics |
| [`src/train`](src/train) | SciCap fine-tuning scripts |
| [`src/analysis`](src/analysis) | Result archive, plot, and collage utilities |
| [`scripts/slurm`](scripts/slurm) | GPU/SLURM launch scripts with sanitized paths |
| [`configs`](configs) | Reproducible experiment configuration templates |
| [`results`](results) | Published CSV summaries, raw JSON metrics, and training histories |
| [`figures`](figures) | Training plots and selected qualitative examples |
| [`docs`](docs) | Methodology, dataset, reproducibility, result notes, and GitHub Pages page |

## Reproducibility

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build appropriate for your GPU/cluster if you plan to reproduce the full experiments. The thesis runs used Python 3.10.19, PyTorch 2.7.1+cu118, Transformers 4.45.2, and an NVIDIA H200 GPU.

Prepare external data:

```bash
# Place Flickr30K files under data/flickr30k/
# Place the SciCap Hugging Face snapshot files under data/scicap_source/
python -m src.prepare_scicap_split --snapshot data/scicap_source
```

Run examples:

```bash
python -m src.eval.evaluate_clip --config configs/experiment_config.yaml --dataset flickr30k --split test
python -m src.eval.evaluate_blip2 --config configs/experiment_config.yaml --dataset scicap --split test --rerank-top-k 128
python -m src.analysis.plot_training_curves
```

Full reproduction requires the external datasets, official model downloads, GPU memory suitable for BLIP-2, and, for the fine-tuned numbers, training from the provided scripts or equivalent checkpoints. Model weights are downloaded from their official packages or Hugging Face when running the code.

## Academic And Legal Notes

This repository is for academic reproducibility. Datasets are not redistributed and must be obtained from their official sources. Model weights and checkpoints are not included and remain governed by their original licenses. The small qualitative collages are included only as selected retrieval examples, not as dataset redistribution.

The MIT license applies to the original source code in this repository. It does not grant rights over external datasets, model weights, benchmark images, or third-party model code.

