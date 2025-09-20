import os
import time
import cv2
import numpy as np
from mss import mss
import keyboard
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import Optional

class ChromeDinoEnv(Env):
    """
    Vision-only Gymnasium env for Chrome Dino with improved region detection
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        auto_calibrate: bool = True,
        templates_dir: str = "templates",
        monitor_index: int = 1,
        game_region: Optional[dict] = None,
        blur: bool = True,
        hist_eq: bool = True,
        reward_mode: str = "sparse",
        action_sleep: float = 0.10,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__()

        # --- Spaces ---
        self.action_space = Discrete(3)  # 0 noop, 1 jump, 2 duck
        self.observation_space = Box(low=0, high=255, shape=(83, 100, 1), dtype=np.uint8)

        # --- Screen grabs ---
        self.sct = mss()
        
        # Fix for monitor selection - try monitor 0 first, then 1
        if monitor_index == 0 and len(self.sct.monitors) > 1:
            self.monitor = self.sct.monitors[1]  # Monitor 1 is usually the main display
        elif monitor_index < len(self.sct.monitors):
            self.monitor = self.sct.monitors[monitor_index]
        else:
            self.monitor = self.sct.monitors[0]
            
        print(f"Using monitor: {self.monitor}")

        # Default regions (will be calibrated if auto_calibrate=True)
        self.game_region = game_region or {'top': 300, 'left': 100, 'width': 600, 'height': 150}
        self.auto_calibrate = bool(auto_calibrate)

        # --- Templates ---
        self.templates_dir = templates_dir
        self.dino_tmpl = None
        self._load_templates()

        # --- Ablation toggles / runtime knobs ---
        self.use_blur = bool(blur)
        self.use_hist_eq = bool(hist_eq)
        self.reward_mode = reward_mode
        self.action_sleep = float(action_sleep)

        # --- RNG / state ---
        self._rng = np.random.default_rng(seed)
        self.game_over = False
        self.step_count = 0
        self.prev_frame_small = None
        self.static_frames_count = 0

        # Auto-calibrate after everything is initialized
        if self.auto_calibrate:
            try:
                self._calibrate_regions()
            except Exception as e:
                print(f"[WARN] Auto-calibration failed ({e}); using provided defaults.")

    def _load_templates(self):
        """Load templates with error handling"""
        try:
            dino_path = os.path.join(self.templates_dir, "dino.png")            
            if os.path.exists(dino_path):
                self.dino_tmpl = cv2.imread(dino_path, cv2.IMREAD_GRAYSCALE)
                if self.dino_tmpl is not None:
                    print("Dino template loaded successfully")
        except Exception as e:
            print(f"[WARN] Failed to load templates: {e}")

    def _grab_full(self) -> np.ndarray:
        """Grab the full monitor as BGR uint8."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        
        # Handle different channel formats
        if img.shape[2] == 4:  # BGRA
            img = img[:, :, :3]  # Remove alpha channel
        elif img.shape[2] == 3 and img.dtype == np.uint8:
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img.astype(np.uint8)

    def _grab(self, region: dict) -> np.ndarray:
        """Grab a specific region as BGR uint8."""
        # Always use the full screen approach and crop
        # This avoids MSS region-specific issues
        full = self._grab_full()
        
        # Reduce the width by half as suggested - the actual game area is much narrower
        adjusted_width = region['width'] // 2
        
        y1, y2 = region['top'], region['top'] + region['height']
        x1, x2 = region['left'], region['left'] + adjusted_width
        
        # Ensure coordinates are within bounds
        y1 = max(0, min(y1, full.shape[0]))
        y2 = max(0, min(y2, full.shape[0]))
        x1 = max(0, min(x1, full.shape[1]))
        x2 = max(0, min(x2, full.shape[1]))
        
        cropped = full[y1:y2, x1:x2]
                
        return cropped

    def _find_dino_game_region(self, frame_bgr: np.ndarray):
        """Find the Dino game region with mode-specific strategies."""

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.dino_tmpl is not None:
            pt, scale = self._match_template_multi_scale(gray, self.dino_tmpl, thr=0.5)
            if pt is not None:
                dino_x, dino_y = pt
                return {
                    'top': max(0, dino_y - 50),
                    'left': max(0, dino_x - 100),
                    'width': 610,
                    'height': 200
                }
            return None

    def detect_color_mode(self, frame_bgr: np.ndarray) -> str:
        """More accurate color mode detection."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram to understand brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        
        # Calculate metrics
        avg_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # More sophisticated mode detection
        dark_pixels = np.sum(gray < 50) / gray.size
        light_pixels = np.sum(gray > 200) / gray.size
        
        # Light mode: high average brightness, many light pixels
        if avg_brightness > 160 and light_pixels > 0.4:
            return "light"
        # Dark mode: low average brightness, many dark pixels  
        elif avg_brightness < 100 and dark_pixels > 0.4:
            return "dark"
        # Ambiguous case - use edge-based detection
        else:
            return "ambiguous"

    def _match_template_multi_scale(self, gray_img: np.ndarray, tmpl: np.ndarray,
                                scales=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3), thr=0.55):
        """Improved template matching that works in both modes."""
        best = (None, -1.0, 1.0)
        
        # Detect mode and adjust template accordingly
        mode = self.detect_color_mode(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR))
        
        if mode == "light":
            # In light mode, invert the template to match dark dino on white background
            tmpl_processed = 255 - tmpl
        else:
            # In dark mode, use the template as-is (light dino on dark background)
            tmpl_processed = tmpl
        
        tmpl_norm = cv2.normalize(tmpl_processed, None, 0, 255, cv2.NORM_MINMAX)
        
        for s in scales:
            try:
                tw = max(10, int(tmpl.shape[1] * s))
                th = max(10, int(tmpl.shape[0] * s))
                t = cv2.resize(tmpl_norm, (tw, th), interpolation=cv2.INTER_AREA)
                
                if gray_img.shape[0] < t.shape[0] or gray_img.shape[1] < t.shape[1]:
                    continue
                    
                search_norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
                res = cv2.matchTemplate(search_norm, t, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                
                if max_val > best[1]:
                    best = (max_loc, max_val, s)
                    
            except Exception as e:
                continue
                
        pt, score, s = best
        return (pt, s) if score >= thr else (None, None)

    def _calibrate_regions(self):
        """Auto-detect game canvas using multiple strategies."""
        print("Calibrating regions...")
        
        # Make sure game is visible (press space to start if needed)
        keyboard.press_and_release('space')
        time.sleep(0.5)
        
        # Grab multiple frames
        frames = []
        for _ in range(3):
            frames.append(self._grab_full())
            time.sleep(0.1)
        
        # Try each frame until we find the game
        for i, frame in enumerate(frames):
            game_region = self._find_dino_game_region(frame)
            if game_region:
                self.game_region = game_region
                
                # Test the region immediately after detection
                print("Testing detected region...")
                test_grab = self._grab(self.game_region)
                print(f"Test grab successful: {test_grab.shape}")
                return
        
        print(f"[WARN] Could not auto-detect game region. Using default region: {self.game_region}")

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.use_blur:
            gray = cv2.GaussianBlur(gray, (3, 3), 1.5)
        if self.use_hist_eq:
            gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (100, 83), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized.astype(np.uint8), axis=-1)

    def _check_game_over_pixel_diff(self, obs: np.ndarray) -> bool:
        """Detect game over by checking if the screen becomes static (game over screen)"""
        small = obs.squeeze()
        
        if self.prev_frame_small is None:
            self.prev_frame_small = small
            return False
        
        # Calculate pixel difference
        diff = np.abs(small.astype(np.int32) - self.prev_frame_small.astype(np.int32)).sum()
        self.prev_frame_small = small
        
        # If very little change for several frames, probably game over
        if diff < 5000:  # Very small change
            self.static_frames_count += 1
        else:
            self.static_frames_count = 0
        
        # If static for 5 consecutive frames, assume game over
        return self.static_frames_count >= 5

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)

        # Start/restart game
        keyboard.press_and_release('space')
        time.sleep(1.0)

        # Recalibrate if auto_calibrate is enabled
        if self.auto_calibrate:
            try:
                self._calibrate_regions()
            except Exception as e:
                print(f"[WARN] Re-calibration failed: {e}")

        self.game_over = False
        self.step_count = 0
        self.prev_frame_small = None
        self.static_frames_count = 0

        obs = self._preprocess(self._grab(self.game_region))
        info = {}
        return obs, info

    def step(self, action: int):
        if action == 1:
            keyboard.press_and_release('up')
        elif action == 2:
            keyboard.press_and_release('down')

        time.sleep(self.action_sleep)

        # Grab the frame before processing to check for game over
        raw_frame = self._grab(self.game_region)
        obs = self._preprocess(raw_frame)
        self.step_count += 1
        done_flag = self._check_game_over_pixel_diff(obs)
        
        # Reward
        if self.reward_mode == "sparse":
            reward = 1.0 if not done_flag else -10.0
        else:
            reward = 0.1 if not done_flag else -1.0

        terminated = bool(done_flag)
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self._grab(self.game_region)
        obs = self._preprocess(frame).squeeze()
        cv2.imshow('Chrome Dino (obs)', obs)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
    
    def debug_show_regions(self):
        """Show detected regions for debugging."""
        import matplotlib.pyplot as plt
        
        # Grab full screen
        full = self._grab_full()
        
        # Also grab the game region to see what's actually captured
        game_img = self._grab(self.game_region)
        
        # Draw rectangle around game region on full screen
        cv2.rectangle(full, (self.game_region['left'], self.game_region['top']),
                    (self.game_region['left'] + self.game_region['width'], 
                    self.game_region['top'] + self.game_region['height']), (0,255,0), 3)
        
        # Show images
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Full screen with region
        axes[0].imshow(cv2.cvtColor(full, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Full Screen\nGreen=detected game region")
        axes[0].axis('off')
        
        # Game region only
        axes[1].imshow(cv2.cvtColor(game_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Game Region (what the agent sees)\nShape: {game_img.shape}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def manual_set_region(self, region: dict):
        """Manually set the game region if auto-detection fails"""
        self.game_region = region
        print(f"Manually set game_region: {self.game_region}")