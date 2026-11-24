"""
GUI Dataset and DataLoader for Alpha Algorithm.
Supports:
1. Synthetic mode: Generates dynamic procedural GUI mockups on the fly.
2. Hugging Face / JSONL mode: Loads OS-Atlas / ScreenSpot formatted GUI grounding data.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from data.synthetic_generator import generate_synthetic_gui_sample


class GUIDataset(Dataset):
    """
    PyTorch Dataset for GUI vision-action grounding.
    """
    def __init__(
        self,
        mode: str = "synthetic",
        data_path: Optional[str] = None,
        num_synthetic_samples: int = 500,
        image_size: int = 768,
        tokenizer: Any = None,
        max_length: int = 512,
        seed: int = 42
    ):
        super().__init__()
        self.mode = mode
        self.data_path = data_path
        self.num_synthetic_samples = num_synthetic_samples
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed

        self.transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),  # Converts PIL image [0..255] to torch.FloatTensor [0..1]
        ])

        self.samples: List[Dict[str, Any]] = []

        if mode == "huggingface":
            if not data_path or not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            self._load_jsonl_dataset(data_path)
        elif mode == "synthetic":
            # For synthetic mode, we generate samples dynamically in __getitem__ using a fixed seed schedule
            pass
        else:
            raise ValueError(f"Unsupported dataset mode: {mode}. Choose 'synthetic' or 'huggingface'.")

    def _load_jsonl_dataset(self, data_path: str):
        """Loads OS-Atlas / ScreenSpot formatted JSONL dataset."""
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self) -> int:
        if self.mode == "synthetic":
            return self.num_synthetic_samples
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "synthetic":
            # Deterministic seed per index for reproducibility
            sample_data = generate_synthetic_gui_sample(
                width=self.image_size,
                height=self.image_size,
                seed=self.seed + idx
            )
            pil_image = sample_data["image"]
            instruction = sample_data["instruction"]
            target_action = sample_data["target_action"]
            norm_bbox = target_action.get("bbox", [0, 0, 0, 0])
            norm_click = [target_action["x"], target_action["y"]]
        else:
            item = self.samples[idx]
            img_path = item["image"]
            if not os.path.isabs(img_path) and self.data_path:
                img_path = os.path.join(os.path.dirname(self.data_path), img_path)
            pil_image = Image.open(img_path).convert("RGB")
            instruction = item.get("instruction") or item.get("query", "Click the target element")

            # Parse bounding box / click target coordinates
            if "bbox" in item:
                norm_bbox = item["bbox"]  # [x1, y1, x2, y2] in [0..1000]
                norm_click = [
                    int((norm_bbox[0] + norm_bbox[2]) / 2),
                    int((norm_bbox[1] + norm_bbox[3]) / 2)
                ]
            elif "point" in item:
                norm_click = item["point"]  # [x, y] in [0..1000]
                norm_bbox = [norm_click[0] - 10, norm_click[1] - 10, norm_click[0] + 10, norm_click[1] + 10]
            else:
                norm_click = [item.get("x", 500), item.get("y", 500)]
                norm_bbox = [norm_click[0] - 10, norm_click[1] - 10, norm_click[0] + 10, norm_click[1] + 10]

            target_action = {
                "action": "click",
                "x": norm_click[0],
                "y": norm_click[1],
                "bbox": norm_bbox,
                "reason": item.get("reason", f"Target matching: {instruction}")
            }

        # Preprocess visual screenshot
        pixel_values = self.transform(pil_image)  # (3, H, W)

        # Build structured prompt and target assistant response
        target_json_str = json.dumps({
            "action": target_action["action"],
            "x": target_action["x"],
            "y": target_action["y"],
            "reason": target_action["reason"]
        }, ensure_ascii=False)

        system_prompt = (
            "<|im_start|>system\n"
            "You are Alpha Algorithm, a GUI vision-action grounding model. Given a desktop screenshot "
            "and user request, return the exact JSON action with normalized coordinates [0..1000]: "
            '{"action": "click|move|type|scroll", "x": int, "y": int, "text": str, "reason": str}<|im_end|>\n'
        )
        user_prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n"
        assistant_prompt = f"<|im_start|>assistant\n{target_json_str}<|im_end|>"

        full_prompt = system_prompt + user_prompt + assistant_prompt
        prefix_prompt = system_prompt + user_prompt + "<|im_start|>assistant\n"

        if self.tokenizer is not None:
            full_encoded = self.tokenizer(
                full_prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            prefix_encoded = self.tokenizer(
                prefix_prompt,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )

            input_ids = full_encoded.input_ids.squeeze(0)
            attention_mask = full_encoded.attention_mask.squeeze(0)

            # Build label mask: only compute loss on assistant tokens
            prefix_len = prefix_encoded.input_ids.shape[1]
            labels = input_ids.clone()
            labels[:prefix_len] = -100  # Mask out system and user tokens
        else:
            # Fallback dummy tensors if tokenizer is not passed (e.g. mock test)
            input_ids = torch.zeros(16, dtype=torch.long)
            attention_mask = torch.ones(16, dtype=torch.long)
            labels = torch.zeros(16, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "instruction": instruction,
            "target_action": target_action,
            "norm_bbox": norm_bbox,
            "norm_click": norm_click
        }


def gui_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to pad variable-length token sequences in batch.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

    # Determine max sequence length in this batch
    max_len = max(item["input_ids"].shape[0] for item in batch)
    b_size = len(batch)

    padded_input_ids = torch.full((b_size, max_len), fill_value=0, dtype=torch.long)
    padded_attention_mask = torch.zeros((b_size, max_len), dtype=torch.long)
    padded_labels = torch.full((b_size, max_len), fill_value=-100, dtype=torch.long)

    instructions = []
    target_actions = []
    norm_bboxes = []
    norm_clicks = []

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        padded_input_ids[i, :seq_len] = item["input_ids"]
        padded_attention_mask[i, :seq_len] = item["attention_mask"]
        padded_labels[i, :seq_len] = item["labels"]

        instructions.append(item["instruction"])
        target_actions.append(item["target_action"])
        norm_bboxes.append(item["norm_bbox"])
        norm_clicks.append(item["norm_click"])

    return {
        "pixel_values": pixel_values,
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
        "instructions": instructions,
        "target_actions": target_actions,
        "norm_bboxes": norm_bboxes,
        "norm_clicks": norm_clicks
    }


def create_gui_dataloader(
    mode: str = "synthetic",
    data_path: Optional[str] = None,
    num_samples: int = 500,
    batch_size: int = 1,
    image_size: int = 768,
    tokenizer: Any = None,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Helper to instantiate GUIDataset and DataLoader."""
    dataset = GUIDataset(
        mode=mode,
        data_path=data_path,
        num_synthetic_samples=num_samples,
        image_size=image_size,
        tokenizer=tokenizer
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=gui_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
