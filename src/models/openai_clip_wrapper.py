"""
Wrapper pre oficiálnu implementáciu OpenAI CLIP použitú v experimentoch.

Modul realizuje cestu CLIP ViT-L/14 pre úlohu Flickr30K a SciCap
image-text retrieval. Zámerne používa balík `clip` od OpenAI, aby bol
verejný repozitár konzistentný s experimentmi v bakalárskej práci.

Vstupom sú cesty k obrázkom, textové popisy a konfigurované veľkosti
dávok. Výstupom sú normalizované obrazové a textové embeddingy, ktoré sa
neskôr porovnávajú skalárnym súčinom. Po normalizácii je tento súčin
ekvivalentný kosínusovej podobnosti, teda základnému skóre pre retrieval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .base import ModelLoadInfo


class OfficialOpenAIClipWrapper:
    """
    Načíta model CLIP a poskytuje dávkové funkcie na výpočet embeddingov.

    Trieda oba modálne vstupy spracuje nezávisle: obrázky cez obrazový
    enkóder a texty cez textový enkóder. Vďaka tomu možno všetky embeddingy
    vypočítať raz a následne vytvoriť celú maticu podobností jednoduchým
    násobením matíc.
    """

    def __init__(self, model_name: str, device: str = "cuda", precision: str = "fp16") -> None:
        try:
            import clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Official OpenAI CLIP is required. Install with: "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self.clip = clip
        self.torch = torch
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        self.precision = precision if self.device == "cuda" else "fp32"
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        self.model.eval()

    @property
    def info(self) -> ModelLoadInfo:
        """Vráti metadáta uložené spolu so surovými JSON výsledkami."""
        return ModelLoadInfo(
            implementation="official_openai_clip",
            model_name=self.model_name,
            device=self.device,
            precision=self.precision,
        )

    def encode_images(self, image_paths: Iterable[str], batch_size: int = 64):
        """
        Zakóduje obrázky do normalizovaných CLIP obrazových embeddingov.

        Každý obrázok sa otvorí ako RGB, prejde oficiálnym CLIP
        preprocessom a následne obrazovým enkóderom. Výsledné vektory sa
        normalizujú, aby sa dali priamo použiť pri výpočte kosínusovej
        podobnosti s textovými embeddingmi.
        """
        from PIL import Image

        features = []
        image_paths = list(image_paths)
        with self.torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                images = []
                for path in batch_paths:
                    image = Image.open(Path(path)).convert("RGB")
                    images.append(self.preprocess(image))
                tensor = self.torch.stack(images).to(self.device)
                batch_features = self.model.encode_image(tensor)
                # Normalizácia zabezpečí, že neskorší skalárny súčin je
                # ekvivalentný kosínusovej podobnosti používanej pri CLIP.
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                features.append(batch_features.float().cpu())
        return self.torch.cat(features, dim=0)

    def encode_texts(self, texts: List[str], batch_size: int = 256):
        """
        Zakóduje popisy do normalizovaných CLIP textových embeddingov.

        Texty sa tokenizujú oficiálnym tokenizerom CLIP. Parameter
        `truncate=True` je dôležitý najmä pri SciCap, kde môžu byť popisy
        dlhšie ako 77 tokenov podporovaných modelom CLIP.
        """
        features = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                # CLIP má limit 77 tokenov. Explicitné skrátenie robí
                # spracovanie dlhých vedeckých popisov reprodukovateľným.
                tokens = self.clip.tokenize(batch_texts, truncate=True).to(self.device)
                batch_features = self.model.encode_text(tokens)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                features.append(batch_features.float().cpu())
        return self.torch.cat(features, dim=0)
