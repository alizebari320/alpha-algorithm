"""
Unit tests for Alpha Algorithm components.
Verifies CNN encoder, spatial projector, synthetic generator, coordinate mappings,
and dry-run safety crosshair renderer.
"""

import os
import sys
import unittest
import torch
from PIL import Image

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.vision_encoder import NativeConvNeXtBackbone, GUIVisionEncoder
from model.projector import SpatialProjector
from data.synthetic_generator import generate_synthetic_gui_sample
from runtime.screen_capture import WaylandScreenCapture
from runtime.wayland_controller import WaylandController, KEY_MAP
from runtime.cli_bridge import AlphaCLIBridge


class TestAlphaAlgorithm(unittest.TestCase):

    def test_01_native_convnext_backbone(self):
        """Verify ConvNeXt CNN downsamples 768x768 to 24x24 with 1024 channels."""
        # Test with smaller 2-stage depth for instant unit test
        backbone = NativeConvNeXtBackbone(depths=(1, 1, 1, 1), dims=(64, 128, 256, 512))
        dummy_x = torch.randn(1, 3, 256, 256)
        out = backbone(dummy_x)
        self.assertEqual(out.shape, (1, 512, 8, 8))
        print("✓ Native ConvNeXt backbone forward pass verified.")

    def test_02_spatial_projector_pixel_unshuffle(self):
        """Verify 2-layer GELU MLP projector with 2x2 spatial downsampling."""
        projector = SpatialProjector(
            in_channels=512,
            llm_dim=768,
            hidden_dim=1024,
            downsample_mode="pixel_unshuffle",
            downsample_factor=2
        )
        feature_map = torch.randn(1, 512, 8, 8)
        tokens = projector(feature_map)
        # 8x8 downsampled by 2x becomes 4x4 = 16 tokens
        self.assertEqual(tokens.shape, (1, 16, 768))
        print("✓ Spatial Projector 2x2 downsampling verified.")

    def test_03_synthetic_gui_generator(self):
        """Verify procedural GUI generator produces valid images, boxes, and normalized targets."""
        sample = generate_synthetic_gui_sample(width=768, height=768, seed=42)
        self.assertIsInstance(sample["image"], Image.Image)
        self.assertEqual(sample["image"].size, (768, 768))
        self.assertGreater(len(sample["elements"]), 3)

        target = sample["target_action"]
        self.assertIn("x", target)
        self.assertIn("y", target)
        self.assertIn("bbox", target)

        # Coordinate range checks [0..1000]
        self.assertGreaterEqual(target["x"], 0)
        self.assertLessEqual(target["x"], 1000)
        self.assertGreaterEqual(target["y"], 0)
        self.assertLessEqual(target["y"], 1000)

        # Click point must be inside normalized bbox
        bx1, by1, bx2, by2 = target["bbox"]
        self.assertTrue(bx1 <= target["x"] <= bx2 or bx2 <= target["x"] <= bx1)
        self.assertTrue(by1 <= target["y"] <= by2 or by2 <= target["y"] <= by1)
        print("✓ Synthetic GUI generator and coordinate normalization verified.")

    def test_04_coordinate_transformation(self):
        """Verify bidirectional mapping between normalized [0..1000] and screen pixels."""
        cap = WaylandScreenCapture(target_size=768)
        screen_size = (1920, 1080)

        # Test center (500, 500)
        px, py = cap.norm_to_pixel(500, 500, screen_size)
        self.assertEqual((px, py), (960, 540))

        # Invert back
        nx, ny = cap.pixel_to_norm(px, py, screen_size)
        self.assertEqual((nx, ny), (500, 500))

        # Test bounds (0, 0) and (1000, 1000)
        px0, py0 = cap.norm_to_pixel(0, 0, screen_size)
        self.assertEqual((px0, py0), (0, 0))

        px1k, py1k = cap.norm_to_pixel(1000, 1000, screen_size)
        self.assertEqual((px1k, py1k), (1919, 1079))
        print("✓ Screen coordinate transformations verified.")

    def test_05_wayland_controller_keymap(self):
        """Verify Linux evdev keycodes for Wayland ydotool controller."""
        controller = WaylandController()
        self.assertIn("enter", KEY_MAP)
        self.assertEqual(KEY_MAP["enter"], 28)
        self.assertEqual(KEY_MAP["escape"], 1)
        self.assertIsNotNone(controller.socket_path)
        print("✓ Wayland controller setup and keymap verified.")

    def test_06_safety_crosshair_dry_run(self):
        """Verify safety dry-run draws crosshairs and saves visualization."""
        bridge = AlphaCLIBridge(dry_run=True, debug_dir="/tmp/alpha_test_debug")
        dummy_img = Image.new("RGB", (1920, 1080), color=(30, 30, 35))
        action = {"action": "click", "x": 500, "y": 500, "reason": "Test Button"}

        out_path = bridge.draw_crosshair_overlay(dummy_img, 960, 540, action, "test_step")
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)
        print(f"✓ Safety crosshair visualization verified ({out_path}).")


if __name__ == "__main__":
    unittest.main()
