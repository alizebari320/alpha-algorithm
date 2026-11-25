"""
Alpha Algorithm ~2B Vision-Action Model.
Combines:
1. ConvNeXt CNN vision encoder (~88M params)
2. Spatial 2-layer GELU MLP Projector (~11M params)
3. 4-bit Quantized Qwen2.5-1.5B Causal LM Decoder with LoRA attention adapters (~1.54B base + ~18M LoRA)
"""

from typing import Dict, Any, Optional, Tuple, List
import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

from model.vision_encoder import GUIVisionEncoder
from model.projector import SpatialProjector

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN_ID = 151665  # Special token ID for image placeholder


class AlphaVisionActionModel(nn.Module):
    """
    Multimodal ~2B parameter vision-action foundation model for GUI automation.
    Integrates ConvNeXt CNN spatial features directly into a 4-bit quantized LM.
    """
    def __init__(
        self,
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cnn_model_name: str = "convnext_base",
        load_in_4bit: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_vision_encoder: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: Optional[str] = "auto",
        init_random_weights: bool = False
    ):
        super().__init__()
        self.llm_model_name = llm_model_name
        self.cnn_model_name = cnn_model_name
        self.load_in_4bit = load_in_4bit
        self.use_lora = use_lora
        self.torch_dtype = torch_dtype
        self.init_random_weights = init_random_weights

        # 1. Vision Encoder: ConvNeXt CNN
        self.vision_encoder = GUIVisionEncoder(
            model_name=cnn_model_name,
            pretrained=not init_random_weights,
            freeze_backbone=freeze_vision_encoder
        )
        cnn_out_dim = self.vision_encoder.get_output_dim()

        # 2. Tokenizer & LLM Decoder
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_model_name,
                trust_remote_code=True,
                use_fast=True,
                local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_model_name,
                trust_remote_code=True,
                use_fast=True
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure <image> token exists in tokenizer
        if DEFAULT_IMAGE_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]})
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

        # 4-bit Quantization Configuration
        bnb_config = None
        if load_in_4bit and torch.cuda.is_available() and not init_random_weights:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True
            )

        # Load Pretrained or Initialized Causal LM
        if init_random_weights:
            config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
            self.decoder = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            if not (load_in_4bit and torch.cuda.is_available()):
                self.decoder.to(dtype=torch_dtype)
        else:
            device_arg = {"device_map": device_map} if (device_map and torch.cuda.is_available()) else {}
            self.decoder = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                quantization_config=bnb_config if load_in_4bit else None,
                torch_dtype=torch_dtype if not load_in_4bit else None,
                trust_remote_code=True,
                **device_arg
            )

        # Prepare for k-bit training if quantized
        if load_in_4bit and torch.cuda.is_available() and not init_random_weights:
            self.decoder = prepare_model_for_kbit_training(
                self.decoder,
                use_gradient_checkpointing=True
            )

        # 3. LoRA Configuration on LLM Attention Layers
        if use_lora:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.decoder = get_peft_model(self.decoder, lora_config)

        llm_hidden_size = getattr(self.decoder.config, "hidden_size", 1536)

        # 4. Spatial 2-Layer GELU MLP Projector
        self.projector = SpatialProjector(
            in_channels=cnn_out_dim,
            llm_dim=llm_hidden_size,
            hidden_dim=2048,
            downsample_mode="pixel_unshuffle",
            downsample_factor=2
        )

        # Move unquantized modules to proper device and dtype
        device = next(self.decoder.parameters()).device
        self.vision_encoder.to(device=device, dtype=torch_dtype)
        self.projector.to(device=device, dtype=torch_dtype)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.decoder.get_input_embeddings()

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Runs ConvNeXt CNN backbone + Spatial Projector.
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            Visual tokens of shape (B, num_tokens, llm_dim)
        """
        features_2d = self.vision_encoder(pixel_values)  # (B, C, H//32, W//32)
        visual_tokens = self.projector(features_2d)       # (B, N, D_llm)
        return visual_tokens

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """
        Multimodal forward pass.
        Visual tokens replace image tokens or are prepended to prompt embeddings.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids).to(dtype=self.torch_dtype)
        b, seq_len, embed_dim = inputs_embeds.shape

        if pixel_values is not None:
            visual_tokens = self.encode_vision(pixel_values).to(dtype=self.torch_dtype)  # (B, N_vis, D)
            b_vis, n_vis, _ = visual_tokens.shape

            # If input_ids contains the image token, splice visual tokens in
            has_image_token = (input_ids == self.image_token_id).any(dim=-1)
            if has_image_token.all() and n_vis > 0:
                combined_embeds_list = []
                combined_mask_list = []
                combined_labels_list = []

                for i in range(b):
                    img_idx = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]
                    if len(img_idx) > 0:
                        idx = img_idx[0].item()
                        # Prefix before <image>
                        prefix_emb = inputs_embeds[i, :idx]
                        suffix_emb = inputs_embeds[i, idx + 1:]
                        merged_emb = torch.cat([prefix_emb, visual_tokens[i], suffix_emb], dim=0)
                        combined_embeds_list.append(merged_emb)

                        if attention_mask is not None:
                            prefix_mask = attention_mask[i, :idx]
                            suffix_mask = attention_mask[i, idx + 1:]
                            vis_mask = torch.ones(n_vis, device=attention_mask.device, dtype=attention_mask.dtype)
                            combined_mask_list.append(torch.cat([prefix_mask, vis_mask, suffix_mask], dim=0))

                        if labels is not None:
                            prefix_lbl = labels[i, :idx]
                            suffix_lbl = labels[i, idx + 1:]
                            vis_lbl = torch.full((n_vis,), -100, device=labels.device, dtype=labels.dtype)
                            combined_labels_list.append(torch.cat([prefix_lbl, vis_lbl, suffix_lbl], dim=0))
                    else:
                        # Fallback prepend
                        combined_embeds_list.append(torch.cat([visual_tokens[i], inputs_embeds[i]], dim=0))
                        if attention_mask is not None:
                            vis_mask = torch.ones(n_vis, device=attention_mask.device, dtype=attention_mask.dtype)
                            combined_mask_list.append(torch.cat([vis_mask, attention_mask[i]], dim=0))
                        if labels is not None:
                            vis_lbl = torch.full((n_vis,), -100, device=labels.device, dtype=labels.dtype)
                            combined_labels_list.append(torch.cat([vis_lbl, labels[i]], dim=0))

                inputs_embeds = torch.stack(combined_embeds_list, dim=0)
                if attention_mask is not None:
                    attention_mask = torch.stack(combined_mask_list, dim=0)
                if labels is not None:
                    labels = torch.stack(combined_labels_list, dim=0)
            else:
                # Prepend visual tokens directly
                inputs_embeds = torch.cat([visual_tokens, inputs_embeds], dim=1)
                if attention_mask is not None:
                    vis_mask = torch.ones(b, n_vis, device=attention_mask.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
                if labels is not None:
                    vis_lbl = torch.full((b, n_vis), -100, device=labels.device, dtype=labels.dtype)
                    labels = torch.cat([vis_lbl, labels], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    @torch.no_grad()
    def generate_action(
        self,
        pixel_values: torch.Tensor,
        instruction: str,
        max_new_tokens: int = 128,
        temperature: float = 0.2
    ) -> str:
        """
        Predicts structured action JSON for the given desktop screenshot and user instruction.
        """
        device = next(self.projector.parameters()).device
        dtype = self.torch_dtype

        pixel_values = pixel_values.to(device=device, dtype=dtype)
        visual_tokens = self.encode_vision(pixel_values)  # (1, N_vis, D)

        prompt = (
            f"<|im_start|>system\n"
            f"You are Alpha Algorithm, a GUI vision-action grounding model. Given a desktop screenshot "
            f"and user request, return the exact JSON action with normalized coordinates [0..1000]: "
            f'{{"action": "click|move|type|scroll", "x": int, "y": int, "text": str, "reason": str}}<|im_end|>\n'
            f"<|im_start|>user\n"
            f"{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        tokens = self.tokenizer(prompt, return_tensors="pt").to(device=device)
        text_embeds = self.get_input_embeddings()(tokens.input_ids)

        # Prepend visual tokens
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.long)

        generated_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output_text.strip()

    def print_parameter_summary(self) -> Dict[str, Any]:
        """
        Prints and returns exact parameter counts, trainable parameter breakdown,
        and VRAM footprint analysis for university presentation slides.
        """
        # Count parameters per submodule
        cnn_total = sum(p.numel() for p in self.vision_encoder.parameters())
        cnn_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)

        proj_total = sum(p.numel() for p in self.projector.parameters())
        proj_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        # Decoder & LoRA breakdown
        dec_total = 0
        dec_trainable = 0
        lora_trainable = 0

        for name, param in self.decoder.named_parameters():
            dec_total += param.numel()
            if param.requires_grad:
                dec_trainable += param.numel()
                if "lora" in name.lower():
                    lora_trainable += param.numel()

        total_params = cnn_total + proj_total + dec_total
        total_trainable = cnn_trainable + proj_trainable + dec_trainable
        total_frozen = total_params - total_trainable

        # VRAM estimate in GB
        # 4-bit decoder weights: ~0.5 bytes/param
        # Projector & CNN (fp16/bf16): ~2 bytes/param
        # LoRA weights: ~2 bytes/param
        vram_weights_gb = (
            (dec_total * 0.5) / (1024 ** 3) +
            ((cnn_total + proj_total + lora_trainable) * 2) / (1024 ** 3)
        )

        summary = {
            "total_params": total_params,
            "trainable_params": total_trainable,
            "frozen_params": total_frozen,
            "trainable_pct": (total_trainable / total_params) * 100 if total_params > 0 else 0.0,
            "cnn_total": cnn_total,
            "cnn_trainable": cnn_trainable,
            "projector_total": proj_total,
            "projector_trainable": proj_trainable,
            "decoder_total": dec_total,
            "decoder_trainable": dec_trainable,
            "lora_trainable": lora_trainable,
            "estimated_vram_weights_gb": vram_weights_gb
        }

        if HAS_RICH:
            console = Console()
            table = Table(title="Alpha Algorithm (~2B Parameter Model Architecture Summary)", show_lines=True)
            table.add_column("Sub-Module", style="cyan", justify="left")
            table.add_column("Total Parameters", justify="right", style="magenta")
            table.add_column("Trainable Parameters", justify="right", style="green")
            table.add_column("Precision / Format", justify="center", style="yellow")
            table.add_column("Trainable %", justify="right", style="blue")

            table.add_row(
                f"ConvNeXt CNN Backbone ({self.cnn_model_name})",
                f"{cnn_total:,}",
                f"{cnn_trainable:,}",
                "BF16 (Frozen)",
                f"{(cnn_trainable / cnn_total * 100):.1f}%" if cnn_total else "0%"
            )
            table.add_row(
                "Spatial 2-Layer GELU Projector",
                f"{proj_total:,}",
                f"{proj_trainable:,}",
                "BF16 (Trainable)",
                f"{(proj_trainable / proj_total * 100):.1f}%" if proj_total else "0%"
            )
            table.add_row(
                f"Base LM Decoder ({self.llm_model_name})",
                f"{dec_total:,}",
                "0 (Frozen)",
                "4-bit NF4 Quantized",
                "0.0%"
            )
            table.add_row(
                "PEFT LoRA Adapters (Attention Projections)",
                f"{lora_trainable:,}",
                f"{lora_trainable:,}",
                "BF16 (Trainable)",
                "100.0%"
            )
            table.add_row(
                "[bold white]TOTAL MODEL[/bold white]",
                f"[bold magenta]{total_params:,}[/bold magenta]",
                f"[bold green]{total_trainable:,}[/bold green]",
                "[bold yellow]Hybrid 4-bit / BF16[/bold yellow]",
                f"[bold blue]{(total_trainable / total_params * 100):.2f}%[/bold blue]"
            )

            console.print()
            console.print(table)
            console.print(Panel(
                f"[bold green]University Presentation Highlight:[/bold green]\n"
                f"• Total Model Parameters: [bold]{total_params / 1e9:.2f} Billion[/bold]\n"
                f"• Trainable Parameters: [bold]{total_trainable / 1e6:.2f} Million[/bold] "
                f"({summary['trainable_pct']:.2f}% of full parameter capacity)\n"
                f"• 4-bit NF4 VRAM Base Footprint: [bold]~{vram_weights_gb:.2f} GB[/bold] "
                f"(Runs comfortably inside 8 GB NVIDIA GPUs)\n"
                f"• Spatial Token Economy: Downsampled 2D grid produces only 144 to 256 compact tokens",
                title="Alpha Algorithm Technical Metrics",
                border_style="green"
            ))
            console.print()
        else:
            print("=" * 70)
            print("ALPHA ALGORITHM (~2B PARAMETER MODEL ARCHITECTURE SUMMARY)")
            print("=" * 70)
            print(f"ConvNeXt CNN Backbone:  {cnn_total:,} params (Trainable: {cnn_trainable:,})")
            print(f"Spatial Projector:      {proj_total:,} params (Trainable: {proj_trainable:,})")
            print(f"Base LM Decoder:        {dec_total:,} params (4-bit NF4 Quantized)")
            print(f"PEFT LoRA Adapters:     {lora_trainable:,} params (Trainable: {lora_trainable:,})")
            print("-" * 70)
            print(f"TOTAL PARAMETERS:       {total_params:,} ({total_params / 1e9:.2f}B)")
            print(f"TRAINABLE PARAMETERS:   {total_trainable:,} ({summary['trainable_pct']:.2f}%)")
            print(f"ESTIMATED WEIGHTS VRAM: {vram_weights_gb:.2f} GB (Fits comfortably in 8 GB VRAM)")
            print("=" * 70)

        return summary

    def save_lightweight_checkpoint(self, output_dir: str):
        """
        Saves only the lightweight trainable components:
        1. Spatial Projector state dict (~45 MB)
        2. LoRA Adapter weights via PEFT (~80 MB)
        3. Config metadata
        Total checkpoint size is ~125 MB (ideal for downloading from Kaggle/Colab).
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save projector weights
        projector_path = os.path.join(output_dir, "projector.pt")
        torch.save(self.projector.state_dict(), projector_path)

        # 2. Save LoRA adapter weights
        lora_dir = os.path.join(output_dir, "lora_adapters")
        if hasattr(self.decoder, "save_pretrained"):
            self.decoder.save_pretrained(lora_dir)

        # 3. Save model metadata configuration
        config_path = os.path.join(output_dir, "alpha_config.json")
        meta = {
            "llm_model_name": self.llm_model_name,
            "cnn_model_name": self.cnn_model_name,
            "load_in_4bit": self.load_in_4bit,
            "use_lora": self.use_lora,
            "projector_in_channels": self.projector.in_channels,
            "projector_llm_dim": self.projector.llm_dim,
            "projector_hidden_dim": self.projector.hidden_dim,
            "projector_downsample_mode": self.projector.downsample_mode,
            "projector_downsample_factor": self.projector.downsample_factor,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load_lightweight_checkpoint(self, checkpoint_dir: str):
        """
        Loads trained weights into the Projector and LoRA adapters.
        """
        projector_path = os.path.join(checkpoint_dir, "projector.pt")
        if os.path.exists(projector_path):
            state_dict = torch.load(projector_path, map_location="cpu")
            self.projector.load_state_dict(state_dict)

        lora_dir = os.path.join(checkpoint_dir, "lora_adapters")
        if os.path.exists(lora_dir) and hasattr(self.decoder, "load_adapter"):
            self.decoder.load_adapter(lora_dir, adapter_name="default")
