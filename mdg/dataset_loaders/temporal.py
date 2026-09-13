from mdg.datamodels import TimeDocument
from typing import List, Optional
from mdg.dataset_loaders import DatasetLoader
from datasets import load_dataset
from mdg.registry import register, DATASET_LOADER


@register(_type=DATASET_LOADER, _name="mock_loader")
class MockDatasetLoader(DatasetLoader):
    def __init__(self, config: dict):
        super().__init__(config)

    def run(self, split: Optional[str], **kwargs) -> dict[str, Optional[List]]:
        return {
            "train": [],
            "dev": None,
            "test": None,
        }


@register(_type=DATASET_LOADER, _name="timex_hf")
class TimexDatasetLoaderHF(DatasetLoader):
    """
    Return time expressions from datasets loaded from HuggingFace
    Args:
        DatasetLoader (_type_): _description_
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = self.config.get("dataset_name", "timex-compositional-sentence")  # e.g., "tempeval3"
        self.dataset_location = self.config.get("hf_location", "mdg-nlp/timex-compositional-sentence")  # e.g., "sagnikrayc/tempeval3"

    def run(self, split: Optional[str] = None) -> dict[str, Optional[List[TimeDocument]]]:
        """
        Load the dataset split (train/validation/test)
        :param split: Dataset split to load
        :return: a dict with keys 'train', 'validation', 'test' and values as lists of documents or None if not available
        """
        if split is not None and split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'validation', 'test' or None.")
        dataset = load_dataset(self.dataset_location, split=split if split is not None else None)
        return {
            "train": [TimeDocument(**doc) for doc in dataset["train"]] if "train" in dataset else None,
            "validation": [TimeDocument(**doc) for doc in dataset["validation"]] if "validation" in dataset else None,
            "test": [TimeDocument(**doc) for doc in dataset["test"]] if "test" in dataset else None,
        }


@register(_type=DATASET_LOADER, _name="tlink_hf")
class TLinkDatasetLoaderHF(DatasetLoader):
    """
    Return TLINK sentence-level classification items from datasets loaded from HuggingFace.
    Each record is a dict like:
      {
        "id": "id77",
        "doc_id": "AFP_ENG_20051217.0286.tml",
        "text": "I <e1>congratulate</e1> ... <e2>said</e2>.",
        "label": "AFTER"
      }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = self.config.get("dataset_name", "tlink-extr-classification-sentence-2-label")
        self.dataset_location = self.config.get(
            "hf_location", "mdg-nlp/tlink-extr-classification-sentence-2-label"  
        )

    def run(self, split: Optional[str] = None) -> dict[str, Optional[List[dict]]]:
        """
        Load the dataset split (train/validation/test)
        :param split: Dataset split to load
        :return: a dict with keys 'train', 'validation', 'test' and values as lists of dicts or None if not available
        """
        if split is not None and split not in ["train", "validation", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be one of 'train', 'validation', 'test' or None."
            )

        dataset = load_dataset(self.dataset_location, split=split if split is not None else None)

        def _norm_label(x: str) -> str:
            return "" if x is None else str(x).strip().upper().replace(" ", "_").replace("-", "_")
        
        KEEP = {"YES", "NO"}

        def _filter_split(name: str):
            if name not in dataset:
                return None
            out = []
            for rec in dataset[name]:
                lbl = _norm_label(rec.get("label"))
                if lbl in KEEP:
                    r = dict(rec)
                    r["label"] = lbl
                    out.append(r)
            return out
        
        return {
            "train": _filter_split("train"),
            "validation": _filter_split("validation"),
            "test": _filter_split("test"),
        }

        
        '''
        return {
            "train": [dict(rec) for rec in dataset["train"]] if "train" in dataset else None,
            "validation": [dict(rec) for rec in dataset["validation"]] if "validation" in dataset else None,
            "test": [dict(rec) for rec in dataset["test"]] if "test" in dataset else None,
        }
        '''

@register(_type=DATASET_LOADER, _name="event_hf")
class EventDatasetLoaderHF(DatasetLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_location = self.config.get("hf_location", "mdg-nlp/domain-eventx-recognition-sentence-updated")

    def run(self, split: Optional[str] = None) -> dict[str, Optional[List[TimeDocument]]]:
        if split is not None and split not in ["train", "validation", "test"]:
            raise ValueError(...)
        dataset = load_dataset(self.dataset_location, split=split if split is not None else None)
        return {
            "train": [TimeDocument(**doc) for doc in dataset["train"]] if "train" in dataset else None,
            "validation": [TimeDocument(**doc) for doc in dataset["validation"]] if "validation" in dataset else None,
            "test": [TimeDocument(**doc) for doc in dataset["test"]] if "test" in dataset else None,
        }
    
@register(_type=DATASET_LOADER, _name="event_hf_3")
class EventDatasetLoaderHF(DatasetLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_location = self.config.get("hf_location", "mdg-nlp/eventx-recognition-document")

    def run(self, split: Optional[str] = None) -> dict[str, Optional[List[TimeDocument]]]:
        if split is not None and split not in ["train", "validation", "test"]:
            raise ValueError(...)
        dataset = load_dataset(self.dataset_location, split=split if split is not None else None)
        return {
            "train": [TimeDocument(**doc) for doc in dataset["train"]] if "train" in dataset else None,
            "validation": [TimeDocument(**doc) for doc in dataset["validation"]] if "validation" in dataset else None,
            "test": [TimeDocument(**doc) for doc in dataset["test"]] if "test" in dataset else None,
        }
    
@register(_type=DATASET_LOADER, _name="event_hf_2")
class EventDatasetLoaderHF(DatasetLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_location = self.config.get("hf_location", "mdg-nlp/eventx-recognition-perturbed")

    def run(self, split: Optional[str] = None) -> dict[str, Optional[List[TimeDocument]]]:
        if split is not None and split not in ["train", "validation", "test"]:
            raise ValueError(...)
        dataset = load_dataset(self.dataset_location, split=split if split is not None else None)
        return {
            "train": [TimeDocument(**doc) for doc in dataset["train"]] if "train" in dataset else None,
            "validation": [TimeDocument(**doc) for doc in dataset["validation"]] if "validation" in dataset else None,
            "test": [TimeDocument(**doc) for doc in dataset["test"]] if "test" in dataset else None,
        }
    

