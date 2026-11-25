"""
Wayland Input Controller for Fedora Linux.
Executes mouse move, click, double-click, scroll, and keyboard typing on Wayland
using ydotool and the uinput subsystem (no X11 dependencies).
"""

from typing import Tuple, Optional
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import shutil
import subprocess


# Mapping of standard keys to Linux evdev keycodes for ydotool
KEY_MAP = {
    "enter": 28,
    "return": 28,
    "esc": 1,
    "escape": 1,
    "backspace": 14,
    "tab": 15,
    "space": 57,
    "left": 105,
    "up": 103,
    "right": 106,
    "down": 108,
    "pageup": 104,
    "pagedown": 109,
    "home": 102,
    "end": 107
}


class WaylandController:
    """
    Wayland-compatible GUI automation controller using ydotool.
    """
    def __init__(self, socket_path: Optional[str] = None):
        self.ydotool_bin = shutil.which("ydotool")
        self.ydotoold_bin = shutil.which("ydotoold")

        # Determine ydotool socket
        default_sock = f"/run/user/{os.getuid()}/.ydotool_socket"
        if socket_path:
            self.socket_path = socket_path
        elif os.environ.get("YDOTOOL_SOCKET"):
            self.socket_path = os.environ["YDOTOOL_SOCKET"]
        elif os.path.exists("/tmp/.ydotool_socket"):
            self.socket_path = "/tmp/.ydotool_socket"
        elif os.path.exists(default_sock):
            self.socket_path = default_sock
        else:
            self.socket_path = "/tmp/.ydotool_socket"

        self.env = os.environ.copy()
        self.env["YDOTOOL_SOCKET"] = self.socket_path

    def check_setup(self) -> Tuple[bool, str]:
        """
        Validates whether ydotool, ydotoold, and /dev/uinput permissions are configured.
        Returns (is_ready, diagnostic_message).
        """
        if not self.ydotool_bin:
            return False, (
                "ydotool is not installed. Install it via Fedora package manager:\n"
                "  sudo dnf install ydotool"
            )

        # Check /dev/uinput access
        uinput_accessible = os.access("/dev/uinput", os.W_OK | os.R_OK)
        if not uinput_accessible:
            return False, (
                "Current user does not have write permissions to /dev/uinput.\n"
                "To fix this permanently on Fedora Wayland, run:\n"
                "  echo 'KERNEL==\"uinput\", GROUP=\"input\", MODE=\"0660\"' | sudo tee /etc/udev/rules.d/80-uinput.rules\n"
                "  sudo usermod -aG input $USER\n"
                "  sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                "  sudo chmod 666 /dev/uinput"
            )

        # Test socket or run test ydotool command
        try:
            res = subprocess.run(
                [self.ydotool_bin, "click", "0x00"],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            if res.returncode == 0:
                return True, "ydotool daemon is active and responsive."
            else:
                err = res.stderr.decode("utf-8", errors="ignore").strip()
                return False, (
                    f"ydotool daemon check returned: {err}\n"
                    "Start the ydotoold daemon on Fedora via:\n"
                    f"  sudo ydotoold --socket-path={self.socket_path} --socket-perm=0666 &\n"
                    f"  export YDOTOOL_SOCKET={self.socket_path}\n"
                    "Or enable as systemd service: systemctl --user enable --now ydotool"
                )
        except Exception as e:
            return False, f"Failed to probe ydotool daemon: {e}"

    def ensure_daemon_started(self) -> bool:
        """
        Attempts to spawn ydotoold in the background if not currently active.
        """
        is_ready, _ = self.check_setup()
        if is_ready:
            return True

        if self.ydotoold_bin and os.access("/dev/uinput", os.W_OK):
            try:
                subprocess.Popen(
                    [self.ydotoold_bin, f"--socket-path={self.socket_path}", "--socket-perm=0666"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(0.5)
                is_ready, _ = self.check_setup()
                return is_ready
            except Exception:
                pass
        return False

    def move_to(self, x: int, y: int) -> bool:
        """
        Moves mouse cursor to absolute pixel coordinates (x, y).
        """
        if not self.ydotool_bin:
            return False
        try:
            # ydotool mousemove -a -x <x> -y <y> or mousemove --absolute <x> <y>
            res = subprocess.run(
                [self.ydotool_bin, "mousemove", "--absolute", str(x), str(y)],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            return res.returncode == 0
        except Exception as e:
            print(f"[WaylandController Error] mousemove: {e}")
            return False

    def click(self, button: str = "left", delay_ms: int = 50) -> bool:
        """
        Performs mouse click.
        button: 'left' (0xC0), 'right' (0xC1), or 'middle' (0xC2).
        """
        if not self.ydotool_bin:
            return False
        button_codes = {
            "left": "0xC0",
            "right": "0xC1",
            "middle": "0xC2"
        }
        btn_hex = button_codes.get(button.lower(), "0xC0")
        try:
            res = subprocess.run(
                [self.ydotool_bin, "click", f"--next-delay={delay_ms}", btn_hex],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            return res.returncode == 0
        except Exception as e:
            print(f"[WaylandController Error] click: {e}")
            return False

    def double_click(self, button: str = "left") -> bool:
        """
        Performs double click.
        """
        s1 = self.click(button=button, delay_ms=30)
        time.sleep(0.08)
        s2 = self.click(button=button, delay_ms=30)
        return s1 and s2

    def scroll(self, steps: int = 3, direction: str = "down") -> bool:
        """
        Scrolls the mouse wheel.
        """
        if not self.ydotool_bin:
            return False
        # In ydotool, mouse wheel movement is delta. Negative is up, positive is down.
        delta = steps if direction.lower() == "down" else -steps
        try:
            # mousemove with wheel delta
            res = subprocess.run(
                [self.ydotool_bin, "mousemove", "-w", str(delta)],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            return res.returncode == 0
        except Exception as e:
            print(f"[WaylandController Error] scroll: {e}")
            return False

    def type_text(self, text: str) -> bool:
        """
        Types a string of text into the active focused window.
        """
        if not self.ydotool_bin:
            return False
        try:
            res = subprocess.run(
                [self.ydotool_bin, "type", "--", text],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            return res.returncode == 0
        except Exception as e:
            print(f"[WaylandController Error] type_text: {e}")
            return False

    def press_key(self, key_name: str) -> bool:
        """
        Presses and releases a key by name (e.g. 'enter', 'tab', 'escape').
        """
        if not self.ydotool_bin:
            return False
        keycode = KEY_MAP.get(key_name.lower())
        if keycode is None:
            print(f"[WaylandController Warning] Unknown key name: {key_name}")
            return False
        try:
            # Format: <keycode>:1 (down) followed by <keycode>:0 (up)
            key_arg = f"{keycode}:1 {keycode}:0"
            res = subprocess.run(
                [self.ydotool_bin, "key", f"{keycode}:1", f"{keycode}:0"],
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2
            )
            return res.returncode == 0
        except Exception as e:
            print(f"[WaylandController Error] press_key: {e}")
            return False
