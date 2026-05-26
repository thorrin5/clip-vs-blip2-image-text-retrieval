# Datasets

## Flickr30K

Flickr30K is used as the standard natural-image retrieval benchmark. The reported evaluation uses the Karpathy test split with 1,000 images and 5,000 captions. Each image has five correct captions, so image-to-text retrieval accepts any of the five captions as correct.

## SciCap

SciCap is used for scientific figure-caption retrieval. The project creates deterministic splits from paired figure-caption records:

| Split | Size |
|---|---:|
| train | 45,000 pairs |
| validation | 1,000 pairs |
| test | 1,000 pairs |

Captions are preprocessed to remove figure-label prefixes and reduce long multi-sentence captions to a shorter shared caption representation. This keeps the comparison fairer for CLIP, whose text encoder has a 77-token context limit.

## License Boundaries

Datasets are not included in this repository. Download them from their official sources and follow their original licenses and terms. The repository license covers only the original code and documentation here.
