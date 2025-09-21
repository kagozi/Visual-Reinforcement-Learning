# # utils/input_backends.py
# import os
# import sys
# import time
# import shutil
# import subprocess

# class KeySender:
#     """
#     Portable key sender with lazy imports and robust fallbacks.

#     Backends:
#       - macOS:   quartz (fast, needs Accessibility) -> osascript (slow, reliable)
#       - Windows: pydirectinput -> pyautogui
#       - Linux:   xdotool (preferred in X11/headless) -> pyautogui

#     Override auto-selection with:  KEYSENDER_BACKEND=<backend>
#     Supported backends: quartz | osascript | pydirectinput | pyautogui | xdotool
#     """

#     def __init__(self, backend: str = "auto"):
#         # allow env override
#         backend = os.environ.get("KEYSENDER_BACKEND", backend)
#         self.backend = self._choose_backend(backend)
#         self._init_backend()

#     def _choose_backend(self, backend: str) -> str:
#         """Pick a sensible default per-OS with clean control flow."""
#         if backend and backend != "auto":
#             return backend

#         plat = sys.platform

#         # macOS first
#         if plat == "darwin":
#             return "quartz"  # will fall back to 'osascript' in _init_backend()

#         # Windows next
#         if plat.startswith("win"):
#             try:
#                 import pydirectinput  # noqa: F401
#                 return "pydirectinput"
#             except Exception:
#                 return "pyautogui"

#         # Linux / anything else
#         if shutil.which("xdotool"):
#             return "xdotool"
#         return "pyautogui"
    
#     def _init_backend(self):
#         b = self.backend
#         self._backend_ready = False

#         try:
#             if b == "quartz":  # macOS
#                 # Requires Accessibility permissions for the Python app
#                 from Quartz import (
#                     CGEventCreateKeyboardEvent,
#                     CGEventPost,
#                     kCGHIDEventTap,
#                 )
#                 self._CGEventCreateKeyboardEvent = CGEventCreateKeyboardEvent
#                 self._CGEventPost = CGEventPost
#                 self._kCGHIDEventTap = kCGHIDEventTap
#                 self._backend_ready = True

#             elif b == "osascript":  # macOS fallback
#                 self._sp = subprocess
#                 self._backend_ready = True

#             elif b == "pydirectinput":  # Windows
#                 import pydirectinput as pdi
#                 pdi.PAUSE = 0
#                 self._pdi = pdi
#                 self._backend_ready = True

#             elif b == "pyautogui":  # Cross-platform (needs a focused window)
#                 import pyautogui as pag
#                 pag.PAUSE = 0
#                 self._pag = pag
#                 self._backend_ready = True

#             elif b == "xdotool":  # Linux/X11 (ideal in Docker/Xvfb)
#                 # quick availability check
#                 self._sp = subprocess
#                 self._sp.run(["xdotool", "--version"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                 self._backend_ready = True

#             else:
#                 raise ValueError(f"Unknown backend: {b}")

#         except Exception as e:
#             # Fallbacks by platform if preferred backend fails to init
#             if sys.platform == "darwin" and b == "quartz":
#                 # fall back to AppleScript automatically
#                 self.backend = "osascript"
#                 self._sp = subprocess
#                 self._backend_ready = True
#             elif sys.platform.startswith("win"):
#                 # Windows: try pyautogui
#                 try:
#                     import pyautogui as pag
#                     pag.PAUSE = 0
#                     self._pag = pag
#                     self.backend = "pyautogui"
#                     self._backend_ready = True
#                 except Exception:
#                     raise RuntimeError(f"Key backend init failed on Windows: {e}")
#             else:
#                 # Linux: try pyautogui
#                 try:
#                     import pyautogui as pag
#                     pag.PAUSE = 0
#                     self._pag = pag
#                     self.backend = "pyautogui"
#                     self._backend_ready = True
#                 except Exception:
#                     raise RuntimeError(f"Key backend init failed on Linux: {e}")

#     # ---------- public API ----------
#     def press(self, key: str, n: int = 1, interval: float = 0.0):
#         """
#         Press a key one or multiple times.

#         Args:
#           key: str, flexible (e.g., "up", "ArrowUp", "UP", "space", " ").
#           n: int, repetitions.
#           interval: seconds between repeats.
#         """
#         if not self._backend_ready:
#             raise RuntimeError("KeySender backend not initialized")

#         norm = self._normalize_key(key)
#         for i in range(max(1, n)):
#             self._press_once(norm)
#             if interval and i < n - 1:
#                 time.sleep(interval)

#     # ---------- per-backend single press ----------
#     def _press_once(self, key: str):
#         b = self.backend
#         if b == "quartz":
#             code = self._quartz_keycode(key)
#             ev_down = self._CGEventCreateKeyboardEvent(None, code, True)
#             self._CGEventPost(self._kCGHIDEventTap, ev_down)
#             ev_up = self._CGEventCreateKeyboardEvent(None, code, False)
#             self._CGEventPost(self._kCGHIDEventTap, ev_up)

#         elif b == "osascript":
#             code = self._mac_keycode(key)
#             # Best-effort; ignore failures so we don't crash training
#             self._sp.run(
#                 ["osascript", "-e", f'tell application "System Events" to key code {code}'],
#                 check=False,
#                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#             )

#         elif b == "pydirectinput":
#             self._pdi.press(self._pdi_keyname(key))

#         elif b == "pyautogui":
#             self._pag.press(self._pag_keyname(key))

#         elif b == "xdotool":
#             self._sp.run(["xdotool", "key", self._xdotool_keyname(key)],
#                          check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#         else:
#             raise ValueError(f"Unknown backend: {b}")

#     # ---------- key normalization & mappings ----------
#     def _normalize_key(self, k: str) -> str:
#         k = (k or "").strip()
#         if not k:
#             return "space"
#         lk = k.lower()

#         # Accept common variants
#         aliases = {
#             "arrowup": "up",
#             "arrowdown": "down",
#             "arrowleft": "left",
#             "arrowright": "right",
#             " ": "space",
#         }
#         lk = aliases.get(lk.replace("_", ""), lk)

#         # Only allow a known set (extend if needed)
#         if lk in {"up", "down", "left", "right", "space"}:
#             return lk
#         # Default to space if unknown to avoid KeyError in training
#         return "space"

#     # Quartz uses *virtual keycodes* (ANSI). These are US layout; good enough for arrow/space.
#     def _quartz_keycode(self, key: str) -> int:
#         # https://eastmanreference.com/complete-list-of-applescript-key-codes
#         mapping = {
#             "up": 0x7E, "down": 0x7D, "left": 0x7B, "right": 0x7C, "space": 0x31
#         }
#         return mapping.get(key, 0x31)

#     # AppleScript key codes (same as Quartz, but numeric Apple codes)
#     def _mac_keycode(self, key: str) -> int:
#         mapping = {
#             "up": 126, "down": 125, "left": 123, "right": 124, "space": 49
#         }
#         return mapping.get(key, 49)

#     # pydirectinput / pyautogui names are already lower-case strings
#     def _pdi_keyname(self, key: str) -> str:
#         return {"space": "space", "up": "up", "down": "down", "left": "left", "right": "right"}[key]

#     def _pag_keyname(self, key: str) -> str:
#         return self._pdi_keyname(key)

#     # xdotool uses capitalized arrows and "space"
#     def _xdotool_keyname(self, key: str) -> str:
#         mapping = {"up": "Up", "down": "Down", "left": "Left", "right": "Right", "space": "space"}
#         return mapping.get(key, "space")

#     # ---------- optional: focus helpers (best-effort) ----------
#     def focus_chromium(self):
#         """
#         Try to focus the Chromium/Chrome window so key presses hit it.
#         No-op if the backend doesn't support it. Safe to call any time.
#         """
#         try:
#             if self.backend == "xdotool":
#                 self._sp.run(["xdotool", "search", "--onlyvisible", "--name", "Chromium"],
#                              check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                 self._sp.run(["xdotool", "windowactivate", "--sync", "$(xdotool getactivewindow)"],
#                              shell=True, check=False,
#                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             elif sys.platform == "darwin":
#                 # AppleScript activate Chrome
#                 self._sp.run(
#                     ["osascript", "-e", 'tell application "Google Chrome" to activate'],
#                     check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
#                 )
#         except Exception:
#             pass


# def open_chrome_dino():
#     """Open Chrome Dino in a portable way and give it focus."""
#     try:
#         if sys.platform == "darwin":
#             subprocess.run(["open","-a","Google Chrome","chrome://dino/"], check=False)
#             # ensure focus
#             subprocess.run(["osascript","-e",'tell app "Google Chrome" to activate'], check=False)
#         elif sys.platform.startswith("win"):
#             subprocess.run(["cmd","/c","start","","chrome","chrome://dino/"], shell=True)
#             # (optional) use pywinauto to focus Chrome window
#         else:
#             subprocess.run(["xdg-open","chrome://dino/"], check=False)
#             # (optional) subprocess.run(["wmctrl","-a","Google Chrome"], check=False)
#     except Exception:
#         pass
#     time.sleep(1.5)
# utils/input_backends.py
import os
import sys
import subprocess
import platform
from typing import Optional

class KeySender:
    """Cross-platform key input handler with Docker support"""
    
    def __init__(self, backend: str = "auto"):
        self.backend = self._select_backend(backend)
        self._setup_backend()
        print(f"Using input backend: {self.backend}")
    
    def _select_backend(self, backend: str) -> str:
        """Select appropriate input backend"""
        if backend != "auto":
            return backend
            
        # Check if we're in Docker/headless environment
        if os.environ.get("DISPLAY") == ":99" or self._is_headless():
            return "xdotool"
        
        # Platform-specific defaults
        system = platform.system().lower()
        if system == "linux":
            return "xdotool" if self._has_xdotool() else "pynput"
        elif system == "darwin":
            return "keyboard"
        elif system == "windows":
            return "keyboard"
        else:
            return "keyboard"  # fallback
    
    def _is_headless(self) -> bool:
        """Check if running in headless environment"""
        return (
            os.environ.get("DISPLAY", "").startswith(":") or
            os.environ.get("SSH_CONNECTION") is not None or
            not os.environ.get("DISPLAY")
        )
    
    def _has_xdotool(self) -> bool:
        """Check if xdotool is available"""
        try:
            subprocess.run(["which", "xdotool"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _setup_backend(self):
        """Initialize the selected backend"""
        if self.backend == "keyboard":
            try:
                import keyboard
                self._keyboard = keyboard
            except ImportError:
                print("Warning: keyboard library not available, falling back to xdotool")
                self.backend = "xdotool"
        
        elif self.backend == "pynput":
            try:
                from pynput.keyboard import Key, Controller
                self._pynput_kb = Controller()
                self._pynput_key = Key
            except ImportError:
                print("Warning: pynput not available, falling back to xdotool")
                self.backend = "xdotool"
        
        elif self.backend == "xdotool":
            if not self._has_xdotool():
                raise RuntimeError("xdotool not found. Install with: apt-get install xdotool")
    
    def press(self, key: str, duration: float = 0.1):
        """Send a key press"""
        try:
            if self.backend == "keyboard":
                self._keyboard.press_and_release(self._convert_key_keyboard(key))
            
            elif self.backend == "pynput":
                pynput_key = self._convert_key_pynput(key)
                self._pynput_kb.press(pynput_key)
                import time
                time.sleep(duration)
                self._pynput_kb.release(pynput_key)
            
            elif self.backend == "xdotool":
                self._send_xdotool_key(key)
            
            else:
                print(f"Warning: Unknown backend {self.backend}")
                
        except Exception as e:
            print(f"Warning: Failed to send key '{key}' with {self.backend}: {e}")
    
    def _convert_key_keyboard(self, key: str) -> str:
        """Convert generic key names to keyboard library format"""
        key_map = {
            "space": "space",
            "up": "up",
            "down": "down", 
            "left": "left",
            "right": "right",
            "enter": "enter"
        }
        return key_map.get(key.lower(), key)
    
    def _convert_key_pynput(self, key: str):
        """Convert generic key names to pynput format"""
        key_map = {
            "space": self._pynput_key.space,
            "up": self._pynput_key.up,
            "down": self._pynput_key.down,
            "left": self._pynput_key.left, 
            "right": self._pynput_key.right,
            "enter": self._pynput_key.enter
        }
        return key_map.get(key.lower(), key)
    
    def _send_xdotool_key(self, key: str):
        """Send key using xdotool (for Docker/headless)"""
        # Convert to xdotool format
        key_map = {
            "space": "space",
            "up": "Up",
            "down": "Down",
            "left": "Left", 
            "right": "Right",
            "enter": "Return"
        }
        
        xdotool_key = key_map.get(key.lower(), key)
        
        try:
            # Find the Chrome window and send key
            subprocess.run([
                "xdotool", 
                "search", "--name", "Chrome",
                "key", xdotool_key
            ], check=False, capture_output=True)
        except subprocess.CalledProcessError as e:
            # Fallback: send to any active window
            try:
                subprocess.run([
                    "xdotool", 
                    "key", xdotool_key
                ], check=False, capture_output=True)
            except Exception:
                print(f"Failed to send key {key} via xdotool")

# For backward compatibility
def get_key_sender(backend: str = "auto") -> KeySender:
    """Factory function to create KeySender instance"""
    return KeySender(backend=backend)