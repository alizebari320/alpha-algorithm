"""
Procedural Synthetic GUI Generator for Alpha Algorithm.
Generates realistic desktop GUI window mockups (buttons, navigation bars,
input fields, icons, title bars) with exact normalized [0..1000] ground-truth
click targets and bounding boxes.
"""

from typing import Dict, Any, List, Tuple
import random
from PIL import Image, ImageDraw, ImageFont


# Palette presets for synthetic Linux desktop environments
PALETTES = [
    {
        "bg": (28, 30, 38),          # Dark theme
        "window_bg": (38, 42, 53),
        "header": (48, 52, 65),
        "text": (240, 240, 245),
        "subtext": (160, 165, 180),
        "btn_primary": (74, 137, 248),
        "btn_secondary": (60, 66, 82),
        "btn_danger": (230, 75, 75),
        "input_bg": (24, 26, 33),
        "border": (70, 75, 95)
    },
    {
        "bg": (245, 246, 250),        # Light theme
        "window_bg": (255, 255, 255),
        "header": (235, 238, 245),
        "text": (30, 35, 45),
        "subtext": (100, 110, 125),
        "btn_primary": (30, 110, 240),
        "btn_secondary": (220, 225, 235),
        "btn_danger": (220, 50, 50),
        "input_bg": (245, 247, 250),
        "border": (210, 215, 225)
    }
]

COMMON_ICONS = ["Terminal", "Files", "Browser", "Settings", "Code", "Music", "Calc", "Help", "System"]
BUTTON_LABELS = ["Submit", "Cancel", "Save", "Apply", "Run", "Download", "Delete", "Next", "Confirm", "Search"]
INPUT_LABELS = ["Search...", "Username", "Password", "URL", "Command", "Filename", "Filter"]


def generate_synthetic_gui_sample(
    width: int = 768,
    height: int = 768,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generates a synthetic desktop GUI screenshot with labeled UI elements.
    Returns:
        Dict containing:
            "image": PIL.Image
            "elements": List[Dict] with element type, text, bbox [x1, y1, x2, y2], normalized [0..1000]
            "target_element": Selected ground-truth target
            "instruction": Natural language command, e.g. "Click the Save button"
            "target_action": Dict with {"action": "click", "x": int, "y": int, "reason": str}
    """
    if seed is not None:
        random.seed(seed)

    palette = random.choice(PALETTES)
    img = Image.new("RGB", (width, height), color=palette["bg"])
    draw = ImageDraw.Draw(img)

    elements: List[Dict[str, Any]] = []

    # Window margin and boundaries
    margin = random.randint(20, 50)
    w_x1, w_y1 = margin, margin
    w_x2, w_y2 = width - margin, height - margin

    # Draw main application window
    draw.rounded_rectangle([w_x1, w_y1, w_x2, w_y2], radius=10, fill=palette["window_bg"], outline=palette["border"], width=1)

    # Window titlebar
    titlebar_h = 36
    draw.rounded_rectangle([w_x1, w_y1, w_x2, w_y1 + titlebar_h], radius=10, fill=palette["header"])
    draw.rectangle([w_x1, w_y1 + 18, w_x2, w_y1 + titlebar_h], fill=palette["header"])

    # Window controls (Close, Minimize, Maximize)
    ctrl_colors = [(255, 95, 87), (254, 188, 46), (40, 201, 64)]
    ctrl_names = ["Close", "Minimize", "Maximize"]
    for i, (col, name) in enumerate(zip(ctrl_colors, ctrl_names)):
        cx = w_x1 + 16 + i * 20
        cy = w_y1 + 18
        r = 6
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        elements.append({
            "type": "window_control",
            "name": f"{name} Window",
            "bbox": [cx - r, cy - r, cx + r, cy + r]
        })

    # Window title text
    app_titles = ["Terminal - Fedora Wayland", "System Settings", "File Manager", "Code Studio", "Alpha GUI Agent"]
    app_title = random.choice(app_titles)
    draw.text((w_x1 + 85, w_y1 + 10), app_title, fill=palette["text"])

    # Sidebar (Navigation icons)
    sidebar_w = 120
    sb_x1, sb_y1 = w_x1, w_y1 + titlebar_h
    sb_x2, sb_y2 = w_x1 + sidebar_w, w_y2
    draw.rectangle([sb_x1, sb_y1, sb_x2, sb_y2], fill=palette["header"], outline=palette["border"], width=1)

    # Sidebar icons
    nav_icons = random.sample(COMMON_ICONS, k=min(5, len(COMMON_ICONS)))
    icon_y = sb_y1 + 20
    for icon_name in nav_icons:
        ix1, iy1 = sb_x1 + 10, icon_y
        ix2, iy2 = sb_x2 - 10, icon_y + 32
        draw.rounded_rectangle([ix1, iy1, ix2, iy2], radius=6, fill=palette["window_bg"], outline=palette["border"], width=1)
        draw.text((ix1 + 12, iy1 + 8), icon_name, fill=palette["text"])
        elements.append({
            "type": "sidebar_icon",
            "name": f"{icon_name} Icon",
            "bbox": [ix1, iy1, ix2, iy2]
        })
        icon_y += 44

    # Main content area
    content_x1 = sb_x2 + 20
    content_y1 = sb_y1 + 20
    content_x2 = w_x2 - 20
    content_y2 = w_y2 - 20

    # Section header / title
    section_title = f"{app_title.split(' - ')[0]} Dashboard"
    draw.text((content_x1, content_y1), section_title, fill=palette["text"])

    # Search / Text Input field
    input_w = random.randint(220, 320)
    input_h = 36
    inp_x1 = content_x1
    inp_y1 = content_y1 + 35
    inp_x2 = inp_x1 + input_w
    inp_y2 = inp_y1 + input_h
    input_placeholder = random.choice(INPUT_LABELS)

    draw.rounded_rectangle([inp_x1, inp_y1, inp_x2, inp_y2], radius=6, fill=palette["input_bg"], outline=palette["border"], width=1)
    draw.text((inp_x1 + 12, inp_y1 + 10), input_placeholder, fill=palette["subtext"])
    elements.append({
        "type": "input",
        "name": f"{input_placeholder} Field",
        "bbox": [inp_x1, inp_y1, inp_x2, inp_y2]
    })

    # Action Buttons Grid
    btns_y = inp_y2 + 40
    selected_buttons = random.sample(BUTTON_LABELS, k=random.randint(4, 7))

    curr_bx = content_x1
    curr_by = btns_y
    btn_h = 38
    btn_w = 110

    for idx, btn_name in enumerate(selected_buttons):
        if curr_bx + btn_w > content_x2:
            curr_bx = content_x1
            curr_by += btn_h + 16

        bx1, by1 = curr_bx, curr_by
        bx2, by2 = bx1 + btn_w, by1 + btn_h

        btn_color = palette["btn_primary"] if idx == 0 else (palette["btn_danger"] if "Delete" in btn_name else palette["btn_secondary"])
        btn_text_color = (255, 255, 255) if idx == 0 or "Delete" in btn_name else palette["text"]

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=6, fill=btn_color)
        draw.text((bx1 + 15, by1 + 11), btn_name, fill=btn_text_color)

        elements.append({
            "type": "button",
            "name": f"{btn_name} Button",
            "bbox": [bx1, by1, bx2, by2]
        })

        curr_bx += btn_w + 16

    # Select one random element as the target ground-truth
    target = random.choice(elements)
    tx1, ty1, tx2, ty2 = target["bbox"]

    # Calculate exact center click point and normalize to [0..1000]
    center_x = (tx1 + tx2) / 2.0
    center_y = (ty1 + ty2) / 2.0

    norm_x = int(round((center_x / width) * 1000))
    norm_y = int(round((center_y / height) * 1000))

    # Normalized bounding box [0..1000]
    norm_bbox = [
        int(round((tx1 / width) * 1000)),
        int(round((ty1 / height) * 1000)),
        int(round((tx2 / width) * 1000)),
        int(round((ty2 / height) * 1000))
    ]

    target_name = target["name"]
    instruction_templates = [
        f"Click the {target_name}",
        f"Please click on {target_name}",
        f"Navigate to and click the {target_name}",
        f"Select {target_name}"
    ]
    instruction = random.choice(instruction_templates)

    target_action = {
        "action": "click",
        "x": norm_x,
        "y": norm_y,
        "bbox": norm_bbox,
        "element": target_name,
        "reason": f"Click {target_name} at normalized coordinates ({norm_x}, {norm_y})"
    }

    return {
        "image": img,
        "elements": elements,
        "target_element": target,
        "instruction": instruction,
        "target_action": target_action,
        "screen_size": (width, height)
    }


def generate_synthetic_batch(batch_size: int = 16, width: int = 768, height: int = 768) -> List[Dict[str, Any]]:
    """Generates a list of synthetic GUI samples."""
    return [generate_synthetic_gui_sample(width, height) for _ in range(batch_size)]
