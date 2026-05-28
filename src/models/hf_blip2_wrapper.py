"""
Wrapper pre BLIP-2 retrieval model z knižnice Hugging Face Transformers.

Modul implementuje model Salesforce/blip2-itm-vit-g použitý pri porovnaní
CLIP vs BLIP-2. BLIP-2 sa v repozitári používa dvojkrokovo: najprv sa
vypočítajú ITC podobnosti pre všetky dvojice obrázok-text a následne sa môže
pre najlepších kandidátov použiť ITM hlava na presnejšie preusporiadanie.

Táto architektúra je dôležitá aj pre interpretáciu výsledkov. ITM
preusporiadanie vie zlepšiť kvalitu párovania, ale je výpočtovo drahšie,
pretože vyžaduje explicitné vyhodnotenie vybraných dvojíc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .base import ModelLoadInfo


class HuggingFaceBlip2RetrievalWrapper:
    """
    Načíta BLIP-2 a poskytuje pomocné metódy pre retrieval experimenty.

    Trieda pokrýva tri hlavné operácie: výpočet obrazových embeddingov,
    výpočet textových embeddingov a skórovanie explicitných párov cez ITM
    hlavu. Vďaka tomu ju používajú zero-shot evaluácie, fine-tuning skripty aj
    generovanie kvalitatívnych príkladov.
    """

    def __init__(self, model_name: str, device: str = "cuda", precision: str = "fp16") -> None:
        try:
            import torch
            from transformers import AutoProcessor
            from transformers import Blip2ForImageTextRetrieval
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face BLIP-2 retrieval requires transformers with "
                "Blip2ForImageTextRetrieval and AutoProcessor."
            ) from exc

        self.torch = torch
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model_name = model_name
        self.precision = precision if self.device == "cuda" else "fp32"
        self.dtype = torch.float16 if self.device == "cuda" and precision == "fp16" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

    @property
    def info(self) -> ModelLoadInfo:
        """Vráti metadáta o použitej implementácii modelu."""
        return ModelLoadInfo(
            implementation="huggingface_transformers",
            model_name=self.model_name,
            device=self.device,
            precision=self.precision,
        )

    def _move_inputs(self, inputs):
        """Presunie vstupy na cieľové zariadenie a nastaví typ obrazových dát."""
        inputs = inputs.to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
        return inputs

    def encode_images(self, image_paths: Iterable[str], batch_size: int = 16):
        """
        Zakóduje obrázky cez vision encoder a Q-Former modelu BLIP-2.

        BLIP-2 reprezentuje každý obrázok viacerými query-token embeddingmi.
        Pri neskoršom ITC porovnávaní sa pre každý text berie najlepšie skóre
        cez tieto query tokeny, čo zodpovedá retrieval receptu modelu.
        """
        from PIL import Image
        import torch.nn.functional as F

        features = []
        image_paths = list(image_paths)
        with self.torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                images = [Image.open(Path(path)).convert("RGB") for path in batch_paths]
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = self._move_inputs(inputs)

                vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                image_embeds = vision_outputs.last_hidden_state
                image_attention_mask = self.torch.ones(
                    image_embeds.size()[:-1],
                    dtype=self.torch.long,
                    device=image_embeds.device,
                )
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                # Každý obrázok má viac query-token reprezentácií; pri ITC
                # skórovaní sa použije najlepšia zhoda s textom.
                query_outputs = self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                image_features = F.normalize(self.model.vision_projection(query_outputs.last_hidden_state), dim=-1)
                features.append(image_features.float().cpu())
        return self.torch.cat(features, dim=0)

    def encode_texts(self, texts: List[str], batch_size: int = 64):
        """
        Zakóduje popisy do textového priestoru používaného BLIP-2 retrievalom.

        Texty sú spracované tokenizerom z rovnakého checkpointu ako model.
        Výstupný textový embedding sa normalizuje, aby bol porovnateľný s
        projekciou obrazových query tokenov.
        """
        import torch.nn.functional as F

        features = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True)
                inputs = self._move_inputs(inputs)

                query_embeds = self.model.embeddings(input_ids=inputs["input_ids"])
                text_outputs = self.model.qformer(
                    query_embeds=query_embeds,
                    query_length=0,
                    attention_mask=inputs.get("attention_mask"),
                    return_dict=True,
                )
                text_features = F.normalize(
                    self.model.text_projection(text_outputs.last_hidden_state[:, 0, :]),
                    dim=-1,
                )
                features.append(text_features.float().cpu())
        return self.torch.cat(features, dim=0)

    def compute_itc_similarity(self, image_features, text_features, image_batch_size: int = 64):
        """
        Vypočíta ITC maticu podobností pre všetky obrázky a texty.

        Riadky matice zodpovedajú obrázkom a stĺpce textovým popisom. Pri
        každom obrázku sa berie maximum cez query tokeny, pretože model môže
        rôznymi tokenmi zachytiť odlišné časti vizuálneho obsahu.
        """
        rows = []
        text_features = text_features.float()
        for start in range(0, image_features.shape[0], image_batch_size):
            image_batch = image_features[start : start + image_batch_size].float()
            # Dávka má pre každý obrázok viac query-token vektorov. Maximum
            # cez query tokeny vyberie najrelevantnejšiu vizuálnu reprezentáciu.
            scores = self.torch.matmul(image_batch, text_features.t()).max(dim=1).values
            rows.append(scores.cpu())
        return self.torch.cat(rows, dim=0)

    def score_pairs(self, image_paths: Iterable[str], texts: Iterable[str], batch_size: int = 8) -> list[float]:
        """
        Ohodnotí explicitné dvojice obrázok-text pomocou ITM hlavy.

        Táto metóda sa používa pri top-k preusporiadaní kandidátov. Na rozdiel
        od ITC matice neporovnáva všetky kombinácie naraz, ale len vybrané
        dvojice, čo je presnejšie, no výrazne pomalšie.
        """
        from PIL import Image

        image_paths = list(image_paths)
        texts = list(texts)
        if len(image_paths) != len(texts):
            raise ValueError("image_paths and texts must have equal length for pair scoring")

        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_paths = image_paths[start : start + batch_size]
                batch_texts = texts[start : start + batch_size]
                images = [Image.open(Path(path)).convert("RGB") for path in batch_paths]
                inputs = self.processor(
                    images=images,
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = self._move_inputs(inputs)
                outputs = self.model(**inputs, use_image_text_matching_head=True)
                score_tensor = self._extract_score_tensor(outputs)
                scores.extend(float(value) for value in score_tensor.detach().float().cpu().reshape(-1).tolist())
        return scores

    def _extract_score_tensor(self, outputs):
        """
        Zjednotí drobné rozdiely vo výstupe rôznych verzií Transformers.

        Knižnica môže ITM skóre vrátiť pod rôznymi názvami atribútov. Funkcia
        preto izoluje tensor skóre a pri dvojtriednom výstupe vyberie skóre
        pozitívnej triedy, teda pravdepodobnosť zhody obrázka s textom.
        """
        candidates = [
            getattr(outputs, "itm_score", None),
            getattr(outputs, "logits", None),
            getattr(outputs, "logits_per_image", None),
        ]
        tensor = next((value for value in candidates if value is not None), None)
        if tensor is None and isinstance(outputs, tuple) and outputs:
            tensor = outputs[0]
        if tensor is None:
            raise RuntimeError("Could not locate a retrieval/ITM score tensor in BLIP-2 outputs")
        if tensor.ndim == 2 and tensor.shape[-1] >= 2:
            return tensor[:, 1]
        return tensor.squeeze()
