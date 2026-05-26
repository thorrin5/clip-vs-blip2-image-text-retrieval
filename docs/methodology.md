# Methodology

The project compares CLIP and BLIP-2 on the same bidirectional image-text retrieval task.

CLIP uses a dual-encoder pipeline. Images and captions are encoded independently with the official OpenAI CLIP `ViT-L/14` model, embeddings are normalized, and retrieval scores are computed by matrix multiplication.

BLIP-2 uses `Salesforce/blip2-itm-vit-g` through Hugging Face Transformers. The evaluation first computes ITC candidate scores and then applies ITM reranking to the top candidates. This gives BLIP-2 a stronger pairwise matching stage but makes evaluation much slower.

All final comparisons use Recall@1, Recall@5, Recall@10, and Mean Recall@K. Mean Recall@K is the average of image-to-text and text-to-image recall for the same K.

SciCap fine-tuning uses deterministic train/validation/test splits with seed 42. Captions are normalized by removing figure prefixes, normalizing whitespace, and selecting a shorter caption span suitable for both models, with CLIP truncation used as a final safety fallback.

