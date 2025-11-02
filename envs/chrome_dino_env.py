import os, sys, time, subprocess
import cv2, numpy as np
import datetime as _dt
from mss import mss
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import Optional
from utils.input_backends import KeySender

class ChromeDinoEnv(Env):
    """
    Vision-only Gymnasium env for Chrome Dino with research ablations.
    Uses KeySender (no 'keyboard' lib). Termination via 'template', 'pixeldiff', or 'either'.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        input_backend: str = "auto",
        auto_calibrate: bool = True,
        templates_dir: str = "templates",
        monitor_index: int = 1,
        game_region: Optional[dict] = None,

        termination_method: str = "template",     # 'pixeldiff'|'template'|'either'
        template_thr: float = 0.62,

        # optional focus on reset (no new tabs)
        focus_on_reset: bool = False,

        # === ABLATIONS ===
        blur: bool = False, 
        hist_eq: bool = False, 
        edge_enhance: bool = False,
        temporal_stack: int = 1,
        obs_resolution: str = "default",  # low|default|high
        obs_channels: str = "grayscale",  # grayscale|rgb|edges|mixed
        reward_mode: str = "sparse", reward_scaling: float = 1.0,
        action_repeat: int = 1, action_sleep: float = 0.10, frame_skip: int = 1,
        noise_level: float = 0.0, brightness_var: float = 0.0, contrast_var: float = 0.0,
        
        debug_dump_dir: Optional[str] = None,   # NEW: where to save PNGs (or None to disable)
        debug_dump_once: bool = True,           # NEW: dump only once per process
        debug_tag: str = "calibration",         # NEW: name tag in filenames


        seed: Optional[int] = None, **kwargs
    ):
        super().__init__()
        self.key = KeySender(backend=input_backend)

        self.ablation_config = {
            'blur': blur, 'hist_eq': hist_eq, 'edge_enhance': edge_enhance,
            'temporal_stack': temporal_stack, 'obs_resolution': obs_resolution,
            'obs_channels': obs_channels, 'reward_mode': reward_mode,
            'reward_scaling': reward_scaling, 'action_repeat': action_repeat,
            'frame_skip': frame_skip, 'noise_level': noise_level,
            'brightness_var': brightness_var, 'contrast_var': contrast_var
        }
        print(f"Ablation config: {self.ablation_config}")

        self.termination_method = termination_method
        self.template_thr = float(template_thr)
        self.focus_on_reset = bool(focus_on_reset)

        # Observation space
        self.obs_resolution = obs_resolution
        self.obs_channels = obs_channels
        self.temporal_stack = max(1, temporal_stack)
        H, W = self._get_obs_dimensions()
        C = self._get_obs_channels()

        self.action_space = Discrete(3)  # 0 noop, 1 jump, 2 duck
        self.observation_space = Box(low=0, high=255, shape=(H, W, C), dtype=np.uint8)
        print(f"Observation space: {self.observation_space.shape}")

        # Capture
        self.sct = mss()
        if monitor_index == 0 and len(self.sct.monitors) > 1:
            self.monitor = self.sct.monitors[1]
        elif monitor_index < len(self.sct.monitors):
            self.monitor = self.sct.monitors[monitor_index]
        else:
            self.monitor = self.sct.monitors[0]

        # Regions/templates
        self.game_region = game_region or {'top': 300, 'left': 100, 'width': 600, 'height': 150}
        self.auto_calibrate = bool(auto_calibrate)
        self.templates_dir = templates_dir
        self.dino_tmpl = None
        self.game_over_tmpl = None
        self._load_templates()

        # Ablation knob states
        self.use_blur = bool(blur); 
        self.use_hist_eq = bool(hist_eq); 
        self.use_edge_enhance = bool(edge_enhance)
        self.reward_mode = reward_mode; 
        self.reward_scaling = float(reward_scaling)
        self.action_repeat = max(1, action_repeat); 
        self.action_sleep = float(action_sleep); 
        self.frame_skip = max(1, frame_skip)
        self.noise_level = max(0.0, noise_level); 
        self.brightness_var = max(0.0, brightness_var); 
        self.contrast_var = max(0.0, contrast_var)

        # State
        self._rng = np.random.default_rng(seed)
        self.game_over = False; self.step_count = 0
        self.prev_frame_small = None; self.static_frames_count = 0
        self.frame_buffer = []; self.last_dino_x = None; self.survival_time = 0

        if self.auto_calibrate:
            try: self._calibrate_regions()
            except Exception as e: print(f"[WARN] Auto-calibration failed ({e}); using defaults.")
        
         # --- debug snapshot config ---
        self.debug_dump_dir = debug_dump_dir
        self.debug_dump_once = bool(debug_dump_once)
        self.debug_tag = str(debug_tag)
        self._debug_dumped = False  # internal guard

    # --------- dims/channels ----------
    def _get_obs_dimensions(self):
        if self.obs_resolution == "low": return (42, 50)
        if self.obs_resolution == "high": return (166, 200)
        return (83, 100)

    def _get_obs_channels(self):
        base = 1 if self.obs_channels == "grayscale" else (4 if self.obs_channels == "mixed" else 3)
        return base * self.temporal_stack

    # --------- templates ----------
    def _load_templates(self):
        try:
            dino_path = os.path.join(self.templates_dir, "dino.png")
            go_path   = os.path.join(self.templates_dir, "game_over.png")
            if os.path.exists(dino_path):
                self.dino_tmpl = cv2.imread(dino_path, cv2.IMREAD_GRAYSCALE)
                if self.dino_tmpl is not None: print("Dino template loaded")
            if os.path.exists(go_path):
                self.game_over_tmpl = cv2.imread(go_path, cv2.IMREAD_GRAYSCALE)
                if self.game_over_tmpl is not None: print("Game Over template loaded")
        except Exception as e:
            print(f"[WARN] Failed to load templates: {e}")

    # --------- grabbing ----------
    def _grab_full(self) -> np.ndarray:
        img = np.array(self.sct.grab(self.monitor))
        if img.shape[2] == 4: img = img[:, :, :3]
        elif img.shape[2] == 3 and img.dtype == np.uint8: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.astype(np.uint8)

    def _grab(self, region: dict) -> np.ndarray:
        full = self._grab_full()
        adj_w = region['width'] // 2  # narrower strip helps for chromedino.com
        y1, y2 = region['top'], region['top'] + region['height']
        x1, x2 = region['left'], region['left'] + adj_w
        y1 = max(0, min(y1, full.shape[0])); y2 = max(0, min(y2, full.shape[0]))
        x1 = max(0, min(x1, full.shape[1])); x2 = max(0, min(x2, full.shape[1]))
        return full[y1:y2, x1:x2]

    # --------- preprocessing ----------
    def _apply_augmentations(self, frame: np.ndarray) -> np.ndarray:
        if self.brightness_var > 0:
            delta = self._rng.normal(0, self.brightness_var * 255)
            frame = np.clip(frame.astype(np.float32) + delta, 0, 255).astype(np.uint8)
        if self.contrast_var > 0:
            factor = self._rng.normal(1.0, self.contrast_var)
            frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return frame

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_bgr = self._apply_augmentations(frame_bgr)
        H, W = self._get_obs_dimensions()

        if self.obs_channels == "grayscale":
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if self.use_blur: g = cv2.GaussianBlur(g, (3,3), 1.5)
            if self.use_hist_eq: g = cv2.equalizeHist(g)
            if self.use_edge_enhance:
                e = cv2.Canny(g, 50, 150); g = cv2.addWeighted(g, 0.8, e, 0.2, 0)
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
            out = np.expand_dims(g, -1)

        elif self.obs_channels == "rgb":
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

        elif self.obs_channels == "edges":
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            e = cv2.Canny(g, 50, 150)
            e = cv2.resize(e, (W, H), interpolation=cv2.INTER_AREA)
            out = np.expand_dims(e, -1)

        else:  # mixed: RGB + edges
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            e = cv2.Canny(g, 50, 150)
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            e = cv2.resize(e, (W, H), interpolation=cv2.INTER_AREA)
            e = np.expand_dims(e, -1)
            out = np.concatenate([rgb, e], axis=-1)

        if self.noise_level > 0:
            noise = self._rng.normal(0, self.noise_level * 255, out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255)
        return out.astype(np.uint8)

    def _handle_temporal_stacking(self, frame: np.ndarray) -> np.ndarray:
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.temporal_stack:
            self.frame_buffer = self.frame_buffer[-self.temporal_stack:]
        while len(self.frame_buffer) < self.temporal_stack:
            self.frame_buffer.insert(0, self.frame_buffer[0] if self.frame_buffer else frame)
        return self.frame_buffer[-1] if self.temporal_stack == 1 else np.concatenate(self.frame_buffer, axis=-1)

    def _ensure_valid_observation(self, obs: np.ndarray) -> np.ndarray:
        if obs.dtype != np.uint8: obs = obs.astype(np.uint8)
        if obs.shape != self.observation_space.shape:
            print(f"WARNING: obs.shape={obs.shape} != {self.observation_space.shape}")
            if obs.size == np.prod(self.observation_space.shape):
                obs = obs.reshape(self.observation_space.shape)
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs

    # --------- termination detection ----------
    def _check_game_over_template(self) -> bool:
        """Robust template matching that works in Docker/headless Chrome"""
        if self.game_over_tmpl is None:
            return False
        
        try:
            # Grab from game region instead of full screen for better reliability
            game_frame = self._grab(self.game_region)
            gray = cv2.cvtColor(game_frame, cv2.COLOR_BGR2GRAY)
            
            # Try multiple thresholds for different environments
            thresholds = [0.45, 0.5, 0.55]  # Lower thresholds for headless
            
            for thr in thresholds:
                pt, score = self._match_template_multi_scale(
                    gray, self.game_over_tmpl, thr=thr
                )
                
                if pt is not None and score > thr:
                    print(f"[TERMINATION] Template match found (score={score:.3f}, thr={thr}) at step {self.step_count}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[WARN] Template matching error: {e}")
            return False
        
    # def _check_game_over_pixeldiff(self, obs: np.ndarray) -> bool:
    #     """Improved pixel difference detection"""
    #     small = np.mean(obs, axis=-1) if (len(obs.shape) == 3 and obs.shape[-1] > 1) else obs.squeeze()
        
    #     if self.prev_frame_small is None:
    #         self.prev_frame_small = small
    #         return False
        
    #     diff = np.abs(small.astype(np.int32) - self.prev_frame_small.astype(np.int32)).sum()
    #     self.prev_frame_small = small
        
    #     # Key insight: Game over screen is STATIC (very low diff)
    #     # But normal gameplay has HIGH diff (dino moving, obstacles scrolling)
    #     # We need CONSECUTIVE static frames to avoid false positives
        
    #     STATIC_THRESHOLD = 3000  # Lower than 5000
    #     CONSECUTIVE_REQUIRED = 3  # Need 3 frames, not 5
        
    #     if diff < STATIC_THRESHOLD:
    #         self.static_frames_count += 1
    #     else:
    #         self.static_frames_count = 0
        
    #     is_game_over = self.static_frames_count >= CONSECUTIVE_REQUIRED
        
    #     if is_game_over:
    #         print(f"[TERMINATION] Pixel diff game over detected at step {self.step_count}")
        
    #     return is_game_over
    
    def _check_game_over_pixeldiff(self, obs: np.ndarray) -> bool:
        """Improved pixel difference detection for headless environments"""
        small = np.mean(obs, axis=-1) if (len(obs.shape) == 3 and obs.shape[-1] > 1) else obs.squeeze()
        
        if self.prev_frame_small is None:
            self.prev_frame_small = small
            return False
        
        diff = np.abs(small.astype(np.int32) - self.prev_frame_small.astype(np.int32)).sum()
        self.prev_frame_small = small
        
        # Adjusted thresholds for headless Chrome
        STATIC_THRESHOLD = 2000  # Even lower for headless
        CONSECUTIVE_REQUIRED = 4  # More conservative
        
        if diff < STATIC_THRESHOLD:
            self.static_frames_count += 1
        else:
            self.static_frames_count = 0
        
        is_game_over = self.static_frames_count >= CONSECUTIVE_REQUIRED
        
        if is_game_over:
            print(f"[TERMINATION] Pixel diff game over detected at step {self.step_count} (diff={diff})")
        
        return is_game_over
    
    def _check_game_over_ocr_fallback(self) -> bool:
        """OCR-based fallback for headless environments"""
        try:
            import pytesseract
            
            # Grab a region where "Game Over" text typically appears
            game_frame = self._grab(self.game_region)
            gray = cv2.cvtColor(game_frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better OCR
            enhanced = cv2.equalizeHist(gray)
            
            # Use pytesseract to detect text
            text = pytesseract.image_to_string(enhanced).lower()
            
            game_over_keywords = ["game over", "over", "game"]
            if any(keyword in text for keyword in game_over_keywords):
                print(f"[TERMINATION] OCR detected game over: '{text.strip()}'")
                return True
                
        except ImportError:
            print("[INFO] pytesseract not available for OCR fallback")
        except Exception as e:
            print(f"[WARN] OCR fallback failed: {e}")
            
        return False
    
    # def _check_combined_termination(self, obs: np.ndarray) -> bool:
    #     """Robust combined termination with safety timeout"""
    #     # Check configured methods
    #     if self.termination_method == "template":
    #         done = self._check_game_over_template()
    #     elif self.termination_method == "pixeldiff":
    #         done = self._check_game_over_pixeldiff(obs)
    #     else:
    #         t1 = self._check_game_over_template()
    #         t2 = self._check_game_over_pixeldiff(obs)
    #         done = t1 or t2
    
    #     return done

    def _check_combined_termination(self, obs: np.ndarray) -> bool:
        """Robust combined termination with multiple fallbacks"""
        # done = self._check_game_over_template()
        done = self._check_game_over_pixeldiff(obs)
        # if self.termination_method == "template":
        #     done = self._check_game_over_template()
        # elif self.termination_method == "pixeldiff":
        #     done = self._check_game_over_pixeldiff(obs)
        # else:  # "either"
        #     t1 = self._check_game_over_template()
        #     t2 = self._check_game_over_pixeldiff(obs)
        #     t3 = self._check_game_over_ocr_fallback()
        #     done = t1 or t2 or t3
        
        # If primary methods fail, try OCR fallback (especially for headless)
        # if not done and self.step_count > 100:  # Only after some gameplay
        #     done = self._check_game_over_ocr_fallback()
            
        # # Safety timeout to prevent infinite episodes
        # if self.step_count >= 5000:  # Very generous timeout
        #     print(f"[TERMINATION] Safety timeout at step {self.step_count}")
        #     return True
            
        return done
    # --------- template helpers ----------
    def detect_color_mode(self, frame_bgr: np.ndarray) -> str:
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        avg = np.mean(g); dark = (g < 50).sum() / g.size; light = (g > 200).sum() / g.size
        if avg > 160 and light > 0.4: return "light"
        if avg < 100 and dark > 0.4: return "dark"
        return "ambiguous"

    def _match_template_multi_scale(self, gray_img: np.ndarray, tmpl: np.ndarray,
                                    scales=(0.7,0.8,0.9,1.0,1.1,1.2,1.3), thr=0.55):
        best = (None, -1.0, 1.0)
        mode = self.detect_color_mode(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR))
        tmpl_proc = 255 - tmpl if mode == "light" else tmpl
        tmpl_norm = cv2.normalize(tmpl_proc, None, 0, 255, cv2.NORM_MINMAX)
        for s in scales:
            try:
                tw = max(10, int(tmpl.shape[1]*s)); th = max(10, int(tmpl.shape[0]*s))
                t = cv2.resize(tmpl_norm, (tw, th), interpolation=cv2.INTER_AREA)
                if gray_img.shape[0] < th or gray_img.shape[1] < tw: continue
                search = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
                res = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > best[1]:
                    best = (max_loc, max_val, s)
            except Exception:
                pass
        pt, score, _ = best
        return (pt, score) if score >= thr else (None, None)

    # --------- calibration ----------
    def _find_dino_game_region(self, frame_bgr: np.ndarray):
        if self.dino_tmpl is None: return None
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        pt, _ = self._match_template_multi_scale(gray, self.dino_tmpl, thr=0.5)
        if pt is None: return None
        dino_x, dino_y = pt
        return {'top': max(0, dino_y-50), 'left': max(0, dino_x-100), 'width': 610, 'height': 200}

    def _calibrate_regions(self):
        print("Calibrating regions…")
        self.key.press("space"); time.sleep(0.5)
        frames = []
        for _ in range(3):
            frames.append(self._grab_full()); time.sleep(0.1)
        for i, frame in enumerate(frames):
            reg = self._find_dino_game_region(frame)
            if reg:
                self.game_region = reg
                print(f"Found game region in frame {i+1}: {self.game_region}")
                return
        print(f"[WARN] Could not auto-detect; using default: {self.game_region}")
        try:
            self._save_debug_regions(when=self.debug_tag or "calibration")
        except Exception as e:
            print(f"[WARN] debug snapshot failed: {e}")

    # --------- gym API ----------
    def seed(self, seed: Optional[int] = None): self._rng = np.random.default_rng(seed)

    def _focus_chrome(self):
        if not self.focus_on_reset: return
        try:
            if sys.platform.startswith("linux"):
                subprocess.run(["bash", "-lc",
                                "xdotool search --onlyvisible --class 'chromium' | tail -n1 | xargs -I{} xdotool windowactivate {}"],
                               check=False)
            elif sys.platform == "darwin":
                subprocess.run(["osascript", "-e", 'tell application "Google Chrome" to activate'], check=False)
            # Windows would need win32 API or bring to front with pyautogui click; skip for now.
        except Exception:
            pass

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None: self.seed(seed)
        self._focus_chrome()
        self.key.press("space")  # start/restart
        time.sleep(1.0)
        self.key.press("space")  # start/restart
        if self.auto_calibrate:
            try: self._calibrate_regions()
            except Exception as e: print(f"[WARN] Re-calibration failed: {e}")

        self.game_over = False; self.step_count = 0
        self.prev_frame_small = None; self.static_frames_count = 0
        self.frame_buffer = []; self.survival_time = 0

        raw = self._grab(self.game_region)
        obs = self._handle_temporal_stacking(self._preprocess_frame(raw))
        try:
            self._save_debug_regions(when="reset")
        except Exception as e:
            print(f"[WARN] debug snapshot failed: {e}")
        return self._ensure_valid_observation(obs), {}

    # def step(self, action: int):
    #     total_reward = 0.0
        
    #     for _ in range(self.action_repeat):
    #         if action == 1: self.key.press("up")
    #         elif action == 2: self.key.press("down")
    #         time.sleep(self.action_sleep)

    #     done = False
    #     for i in range(self.frame_skip):
    #         raw = self._grab(self.game_region)
    #         obs = self._handle_temporal_stacking(self._preprocess_frame(raw))
    #         self.step_count += 1

    #         # Use the combined termination method
    #         done = self._check_combined_termination(obs)

    #         total_reward += self._calculate_reward(done)
    #         if done: break
    #         if i < self.frame_skip - 1: 
    #             time.sleep(self.action_sleep / self.frame_skip)

    #     obs = self._ensure_valid_observation(obs)
        
    #     # Important: Mark truncated episodes for SB3
    #     truncated = (self.step_count >= 2000) and not done
        
    #     return obs, total_reward, done, truncated, {
    #         "step_count": self.step_count,
    #         "survival_time": self.survival_time,
    #         "ablation_config": self.ablation_config,
    # }
    def step(self, action: int):
        reward = 0.0
        # 0: noop, 1: jump, 2: duck
        if action == 1:
            self.key.press("up")
        elif action == 2:
            self.key.press("down")

        # Let the game advance a tick
        time.sleep(self.action_sleep)

        # Grab and preprocess one fresh frame
        raw = self._grab(self.game_region)
        obs = self._preprocess_frame(raw)
        obs = self._ensure_valid_observation(obs)

        # Book-keeping
        self.step_count += 1

        # Termination (template/pixeldiff/either per your setting)
        done = self._check_combined_termination(obs)

        # Simple reward
        # reward = 1.0 if not done else -10.0
        reward += self._calculate_reward(done)

        info = {
            "step_count": self.step_count,
        }

        # Gymnasium API: (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, info


    def render(self):
        frame = self._grab(self.game_region)
        proc = self._preprocess_frame(frame)
        disp = proc[:, :, 0] if proc.shape[-1] not in (3,4) else (proc if proc.shape[-1]==3 else proc[:,:,:3])
        cv2.imshow('Chrome Dino (obs)', disp); cv2.waitKey(1)

    def close(self): cv2.destroyAllWindows()

    # rewards
    def _calculate_reward(self, terminated: bool) -> float:
        base = 0.0
        if self.reward_mode == "sparse":
            base = 1.0 if not terminated else -10.0
        elif self.reward_mode == "dense":
            base = (0.1 if not terminated else -5.0) + (min(self.step_count * 0.001, 1.0) if not terminated else 0.0)
        elif self.reward_mode == "distance":
            base = (0.1 + (self.step_count * 0.01)) if not terminated else -5.0
        elif self.reward_mode == "survival":
            if not terminated:
                self.survival_time += 1; base = np.log(1 + self.survival_time) * 0.1
            else:
                base = -2.0
        return base * self.reward_scaling

    # debug
    def debug_show_regions(self):
        import matplotlib.pyplot as plt
        full = self._grab_full(); game_img = self._grab(self.game_region)
        cv2.rectangle(full, (self.game_region['left'], self.game_region['top']),
                      (self.game_region['left']+self.game_region['width'], self.game_region['top']+self.game_region['height']),
                      (0,255,0), 3)
        import matplotlib
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        ax[0].imshow(cv2.cvtColor(full, cv2.COLOR_BGR2RGB)); ax[0].set_title("Full Screen (green=game)"); ax[0].axis('off')
        ax[1].imshow(cv2.cvtColor(game_img, cv2.COLOR_BGR2RGB)); ax[1].set_title(f"Game Region {game_img.shape}"); ax[1].axis('off')
        plt.tight_layout(); plt.show()
        
    # ---------- debug snapshot writers ----------
    def _save_debug_regions(self, when: str = "calibration"):
        """
        Save a full-screen shot (with green box), the raw game crop,
        and the preprocessed observation into debug_dump_dir.
        """
        if not self.debug_dump_dir:
            return
        if self._debug_dumped and self.debug_dump_once:
            return

        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        outdir = os.path.abspath(self.debug_dump_dir)
        os.makedirs(outdir, exist_ok=True)

        full = self._grab_full()
        # draw game box in green
        cv2.rectangle(
            full,
            (self.game_region['left'], self.game_region['top']),
            (self.game_region['left'] + self.game_region['width'],
             self.game_region['top'] + self.game_region['height']),
            (0, 255, 0), 3
        )
        cv2.imwrite(os.path.join(outdir, f"{ts}_{when}_full_with_box.png"), full)

        crop = self._grab(self.game_region)
        cv2.imwrite(os.path.join(outdir, f"{ts}_{when}_game_region.png"), crop)

        proc = self._preprocess_frame(crop)
        # displayable: pick a single channel if stacked/mixed
        if proc.ndim == 3:
            if proc.shape[-1] == 1:
                disp = proc[..., 0]
            else:
                # If RGB(+edges), write RGB; if stacked frames, use first channel
                disp = proc[..., :3] if proc.shape[-1] >= 3 else proc[..., 0]
        else:
            disp = proc
        cv2.imwrite(os.path.join(outdir, f"{ts}_{when}_processed.png"), disp)

        self._debug_dumped = True

