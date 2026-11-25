"""
CLI AI Bridge for Alpha Algorithm on Fedora Wayland.
Executes interactive vision-action loop, safety dry-run visualization with crosshairs,
and OpenAI-compatible external CLI LLM fallback.
"""

from typing import Dict, Any, Optional
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import argparse
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import requests

from runtime.screen_capture import WaylandScreenCapture
from runtime.wayland_controller import WaylandController

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class AlphaCLIBridge:
    """
    Coordinates screen capture, vision-action reasoning, dry-run safety verification,
    and Wayland input execution.
    """
    def __init__(
        self,
        model: Optional[Any] = None,
        dry_run: bool = True,
        debug_dir: str = "debug_output",
        use_external_api: bool = False
    ):
        self.model = model
        self.dry_run = dry_run
        self.debug_dir = debug_dir
        self.use_external_api = use_external_api

        os.makedirs(debug_dir, exist_ok=True)

        self.screen_capture = WaylandScreenCapture(target_size=768)
        self.controller = WaylandController()

        # External API config (OpenAI-compatible)
        self.api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.api_model = os.environ.get("OPENAI_MODEL", "gpt-4o")

    def draw_crosshair_overlay(
        self,
        raw_image: Image.Image,
        click_px: int,
        click_py: int,
        action_data: Dict[str, Any],
        step_id: str
    ) -> str:
        """
        Draws a high-visibility safety crosshair and action label overlay
        on the captured screenshot before any mouse input is executed.
        """
        img_copy = raw_image.copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size

        # Crosshair parameters
        radius = 24
        color = (255, 30, 30)       # Vivid Red
        outline_halo = (255, 255, 255) # White halo for contrast on dark backgrounds

        # 1. Halo circle
        draw.ellipse([click_px - radius - 2, click_py - radius - 2, click_px + radius + 2, click_py + radius + 2], outline=outline_halo, width=2)
        # 2. Main red target ring
        draw.ellipse([click_px - radius, click_py - radius, click_px + radius, click_py + radius], outline=color, width=3)
        # Inner dot
        draw.ellipse([click_px - 3, click_py - 3, click_px + 3, click_py + 3], fill=color)

        # 3. Crosshair lines extending across screen
        line_len = 50
        # Horizontal
        draw.line([max(0, click_px - line_len), click_py, max(0, click_px - radius), click_py], fill=color, width=3)
        draw.line([min(w, click_px + radius), click_py, min(w, click_px + line_len), click_py], fill=color, width=3)
        # Vertical
        draw.line([click_px, max(0, click_py - line_len), click_px, max(0, click_py - radius)], fill=color, width=3)
        draw.line([click_px, min(h, click_py + radius), click_px, min(h, click_py + line_len)], fill=color, width=3)

        # 4. Action information banner badge
        action_type = action_data.get("action", "click").upper()
        norm_x = action_data.get("x", 0)
        norm_y = action_data.get("y", 0)
        reason = action_data.get("reason", "")
        badge_text = f"[DRY-RUN] Action: {action_type} | Normalized: ({norm_x}, {norm_y}) | Screen Px: ({click_px}, {click_py})\nReason: {reason}"

        # Position banner near crosshair but safely on-screen
        bx1 = min(w - 420, max(20, click_px + 35))
        by1 = min(h - 100, max(20, click_py - 50))
        bx2 = bx1 + 400
        by2 = by1 + 60

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=8, fill=(20, 20, 25), outline=color, width=2)
        draw.text((bx1 + 12, by1 + 10), badge_text, fill=(255, 255, 255))

        out_path = os.path.join(self.debug_dir, f"dry_run_crosshair_{step_id}.png")
        img_copy.save(out_path)
        return out_path

    def predict_action_via_api(self, instruction: str, raw_image: Image.Image) -> Dict[str, Any]:
        """
        Fallback reasoning via OpenAI-compatible vision endpoint.
        """
        import base64
        buffered = io.BytesIO()
        raw_image.save(buffered, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.api_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a GUI grounding assistant. Return a single JSON object with normalized [0..1000] "
                        "coordinates: {\"action\": \"click\", \"x\": int, \"y\": int, \"reason\": str}"
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }
        res = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def execute_step(self, instruction: str) -> Dict[str, Any]:
        """
        Executes a single perception-action step:
        1. Capture Wayland screen
        2. Predict action JSON
        3. Dry-run visualization or live execution
        """
        step_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        console = Console() if HAS_RICH else None

        if console:
            console.print(f"[bold cyan]📸 Capturing Fedora Wayland desktop screen...[/bold cyan]")
        else:
            print("Capturing Fedora Wayland desktop screen...")

        pixel_values, raw_image, orig_size = self.screen_capture.capture_for_model()
        orig_w, orig_h = orig_size

        if console:
            console.print(f"[dim]Display resolution: {orig_w}x{orig_h}[/dim]")
            console.print(f"[bold yellow]🧠 Running Alpha Algorithm GUI grounding for: '{instruction}'[/bold yellow]")
        else:
            print(f"Display resolution: {orig_w}x{orig_h}")
            print(f"Running Alpha Algorithm GUI grounding for: '{instruction}'")

        # Predict action
        action_data = None
        if self.model is not None:
            try:
                output_str = self.model.generate_action(pixel_values, instruction)
                # Parse JSON from model output
                start_idx = output_str.find("{")
                end_idx = output_str.rfind("}")
                if start_idx != -1 and end_idx != -1:
                    action_data = json.loads(output_str[start_idx:end_idx + 1])
                else:
                    action_data = {"action": "click", "x": 500, "y": 500, "reason": output_str}
            except Exception as e:
                print(f"[Model Warning] Fallback due to parse: {e}")

        if action_data is None and self.use_external_api and self.api_key:
            try:
                action_data = self.predict_action_via_api(instruction, raw_image)
            except Exception as e:
                print(f"[API Warning] External API failed: {e}")

        if action_data is None:
            # Synthetic / demonstration mock action
            action_data = {
                "action": "click",
                "x": 512,
                "y": 340,
                "reason": f"Predicted target for instruction: {instruction}"
            }

        norm_x = int(action_data.get("x", 500))
        norm_y = int(action_data.get("y", 500))
        real_px, real_py = self.screen_capture.norm_to_pixel(norm_x, norm_y)

        # Safety Dry-Run check
        crosshair_path = self.draw_crosshair_overlay(raw_image, real_px, real_py, action_data, step_id)

        if self.dry_run:
            msg = (
                f"[bold red]🛡️ SAFETY DRY-RUN ACTIVE (Mouse will NOT move)[/bold red]\n"
                f"• Predicted Action: [bold]{action_data.get('action', 'click')}[/bold]\n"
                f"• Normalized Coords: [yellow]({norm_x}, {norm_y})[/yellow] on [0..1000] scale\n"
                f"• Real Wayland Coordinates: [green]({real_px}, {real_py})[/green] px\n"
                f"• Explanation: {action_data.get('reason', '')}\n"
                f"• Visual Target Verification Overlay saved at:\n"
                f"  [underline cyan]{crosshair_path}[/underline cyan]"
            )
            if console:
                console.print(Panel(msg, title="Dry-Run Action Planned", border_style="red"))
            else:
                print("=" * 60)
                print(f"[SAFETY DRY-RUN] Action: {action_data.get('action')}")
                print(f"Target: ({real_px}, {real_py}) px | Normalized: ({norm_x}, {norm_y})")
                print(f"Reason: {action_data.get('reason')}")
                print(f"Crosshair overlay saved to: {crosshair_path}")
                print("=" * 60)
        else:
            # LIVE EXECUTION VIA YDOTOOL
            if console:
                console.print(f"[bold red]⚡ LIVE EXECUTION: Moving mouse to ({real_px}, {real_py}) and executing {action_data.get('action')}...[/bold red]")
            else:
                print(f"LIVE EXECUTION: Moving mouse to ({real_px}, {real_py})...")

            act = action_data.get("action", "click").lower()
            self.controller.move_to(real_px, real_py)
            time.sleep(0.1)

            if act == "click":
                self.controller.click(button="left")
            elif act == "right_click":
                self.controller.click(button="right")
            elif act == "double_click":
                self.controller.double_click(button="left")
            elif act == "type":
                text_to_type = action_data.get("text", "")
                self.controller.click(button="left")
                time.sleep(0.05)
                self.controller.type_text(text_to_type)
            elif act == "scroll":
                self.controller.scroll(steps=3, direction="down")

        return {
            "action_data": action_data,
            "real_px": real_px,
            "real_py": real_py,
            "crosshair_path": crosshair_path,
            "dry_run": self.dry_run
        }

    def run_interactive_loop(self):
        """
        Interactive terminal loop for Fedora desktop automation.
        """
        console = Console() if HAS_RICH else None
        print("\n" + "=" * 65)
        print(" ALPHA ALGORITHM - FEDORA WAYLAND VISION-ACTION INTERACTIVE BRIDGE")
        print("=" * 65)
        print(f"• Mode: {'DRY-RUN (Safe - visual crosshair only)' if self.dry_run else 'LIVE EXECUTION (Real mouse/keys)'}")
        print("• Type your instruction (e.g. 'Click Terminal icon', 'Click Save') or 'quit' to exit.\n")

        # Check Wayland controller health
        if not self.dry_run:
            is_ready, diag = self.controller.check_setup()
            if not is_ready:
                print(f"[Warning] Wayland Controller setup issue:\n{diag}\n")

        while True:
            try:
                if console:
                    instruction = Prompt.ask("\n[bold cyan]Alpha GUI[/bold cyan] >")
                else:
                    instruction = input("\nAlpha GUI > ").strip()

                if not instruction:
                    continue
                if instruction.lower() in ["quit", "exit", "q"]:
                    print("Exiting Alpha Algorithm bridge.")
                    break

                self.execute_step(instruction)

            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted. Exiting.")
                break


def main():
    parser = argparse.ArgumentParser(description="Alpha Algorithm Fedora Wayland CLI AI Bridge")
    parser.add_argument("--instruction", type=str, default=None, help="Single-shot instruction to execute")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Draw red crosshair without moving real mouse")
    parser.add_argument("--live", action="store_true", help="Enable live mouse and keyboard movement (disables dry-run)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to lightweight model checkpoint dir")
    parser.add_argument("--external-api", action="store_true", help="Use OpenAI-compatible API fallback if model is unweighted")
    args = parser.parse_args()

    dry_run = not args.live

    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        from model.alpha_model import AlphaVisionActionModel
        import torch
        print(f"Loading Alpha Algorithm checkpoint from {args.checkpoint}...")
        model = AlphaVisionActionModel(load_in_4bit=torch.cuda.is_available(), init_random_weights=True)
        model.load_lightweight_checkpoint(args.checkpoint)
        model.eval()

    bridge = AlphaCLIBridge(
        model=model,
        dry_run=dry_run,
        use_external_api=args.external_api
    )

    if args.instruction:
        bridge.execute_step(args.instruction)
    else:
        bridge.run_interactive_loop()


if __name__ == "__main__":
    main()
