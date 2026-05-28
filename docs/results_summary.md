# Results Summary

The final CSV summaries are stored in [`../results`](../results). The main comparison is [`../results/full_comparison.csv`](../results/full_comparison.csv). CSV timing values are rounded summaries; raw JSON payloads retain full floating-point timing.

| Dataset | Model | Setting | Mean R@1 | Mean R@5 | Mean R@10 | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Flickr30K | CLIP | zero-shot | 75.06 | 92.34 | 95.56 | 5 |
| Flickr30K | BLIP-2 | zero-shot | 91.55 | 98.37 | 99.19 | 4247 |
| SciCap | CLIP | zero-shot | 21.65 | 33.95 | 40.25 | 13 |
| SciCap | BLIP-2 | zero-shot | 26.30 | 35.55 | 39.95 | 1988 |
| SciCap | CLIP | fine-tuned | 33.95 | 50.55 | 58.10 | 10 |
| SciCap | BLIP-2 | fine-tuned | 30.00 | 42.20 | 47.75 | 5195 |

The main pattern is that BLIP-2 is more accurate on Flickr30K and stronger zero-shot on SciCap, but its ITM reranking step is computationally expensive. CLIP is much faster and showed the larger fine-tuning gain on SciCap under the reported setup.
