"""
Wayland Screen Capture for Fedora Linux.
Supports:
1. 'grim' native Wayland compositor screencopy capture
2. XDG Desktop Portal / GNOME Shell / KDE Spectacle DBus capture
3. Fallback dummy / simulated desktop for headless testing
"""

from typing import Tuple, Optional
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import io
import shutil
import subprocess
from PIL import Image
import torch
import torchvision.transforms as T


class WaylandScreenCapture:
    """
    Wayland-native screen grabber for Fedora Linux.
    Captures screenshots and resizes frames for ConvNeXt CNN input while preserving
    original display resolution geometry for precise coordinate transformation.
    """
    def __init__(self, target_size: int = 768):
        self.target_size = target_size
        self.last_orig_size: Tuple[int, int] = (1920, 1080)
        self.transform = T.Compose([
            T.Resize((target_size, target_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        # Discover available screen capture backends
        self.has_grim = shutil.which("grim") is not None
        self.has_gnome = shutil.which("gnome-screenshot") is not None
        self.has_spectacle = shutil.which("spectacle") is not None

    def capture_raw(self) -> Image.Image:
        """
        Captures the primary desktop screen image.
        Tries grim first, then GNOME/KDE tools, then synthetic fallback.
        """
        # 1. Primary Wayland tool: grim
        if self.has_grim:
            try:
                # Capture directly to stdout PNG bytes
                res = subprocess.run(
                    ["grim", "-t", "png", "-"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=4,
                    check=True
                )
                if res.stdout and len(res.stdout) > 100:
                    img = Image.open(io.BytesIO(res.stdout)).convert("RGB")
                    self.last_orig_size = img.size
                    return img
            except Exception:
                pass

        # 2. GNOME Wayland fallback: gnome-screenshot
        if self.has_gnome:
            try:
                tmp_path = "/tmp/alpha_screen_capture.png"
                subprocess.run(
                    ["gnome-screenshot", "-f", tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=True
                )
                if os.path.exists(tmp_path):
                    img = Image.open(tmp_path).convert("RGB")
                    self.last_orig_size = img.size
                    os.remove(tmp_path)
                    return img
            except Exception:
                pass

        # 3. KDE Wayland fallback: spectacle
        if self.has_spectacle:
            try:
                tmp_path = "/tmp/alpha_screen_capture.png"
                subprocess.run(
                    ["spectacle", "-b", "-n", "-o", tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=True
                )
                if os.path.exists(tmp_path):
                    img = Image.open(tmp_path).convert("RGB")
                    self.last_orig_size = img.size
                    os.remove(tmp_path)
                    return img
            except Exception:
                pass

        # 4. Fallback test screen (e.g. headless or permission denied)
        from data.synthetic_generator import generate_synthetic_gui_sample
        sample = generate_synthetic_gui_sample(width=1920, height=1080)
        self.last_orig_size = (1920, 1080)
        return sample["image"]

    def capture_for_model(self) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """
        Captures current desktop screen and formats it for Alpha Algorithm's CNN backbone.
        Returns:
            pixel_values: torch.FloatTensor of shape (1, 3, target_size, target_size)
            raw_image: Original PIL.Image
            orig_size: (width, height) of original screen
        """
        pil_image = self.capture_raw()
        orig_w, orig_h = pil_image.size
        self.last_orig_size = (orig_w, orig_h)

        tensor = self.transform(pil_image)  # (3, H, W)
        pixel_values = tensor.unsqueeze(0)  # (1, 3, H, W)

        return pixel_values, pil_image, (orig_w, orig_h)

    def norm_to_pixel(self, norm_x: int, norm_y: int, screen_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Converts model normalized coordinates [0..1000] to real screen pixel coordinates.
        """
        w, h = screen_size or self.last_orig_size
        px = int(round((norm_x / 1000.0) * w))
        py = int(round((norm_y / 1000.0) * h))
        # Clamp to bounds
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        return px, py

    def pixel_to_norm(self, px: int, py: int, screen_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Converts real screen pixel coordinates to model normalized coordinates [0..1000].
        """
        w, h = screen_size or self.last_orig_size
        norm_x = int(round((px / float(w)) * 1000.0))
        norm_y = int(round((py / float(h)) * 1000.0))
        norm_x = max(0, min(1000, norm_x))
        norm_y = max(0, min(1000, norm_y))
        return norm_x, norm_y
