import os
import time
import cv2
import numpy as np
import pytesseract
from mss import mss
import keyboard
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from typing import Optional, Tuple, Dict
import subprocess
import platform
# Try to import pyautogui as fallback
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

class DynamicChromeDinoEnv(Env):
    """
    Chrome Dino environment with dynamic window detection and region identification.
    Automatically finds Chrome windows and detects the game area.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        detection_method: str = "auto",  # 'window', 'template', 'color', 'auto'
        blur: bool = True,
        hist_eq: bool = True,
        reward_mode: str = "sparse",
        action_sleep: float = 0.10,
        debug: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.action_space = Discrete(3)  # 0 noop, 1 jump, 2 duck
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
        self.sct = mss()
        self.detection_method = detection_method
        self.use_blur = blur
        self.use_hist_eq = hist_eq
        self.reward_mode = reward_mode
        self.action_sleep = action_sleep
        self.debug = debug
        
        # Game state
        self.game_over = False
        self.step_count = 0
        self.game_region = None
        self.chrome_windows = []
        
        # Detection will happen during reset
        print("Environment initialized. Call reset() to detect game region.")

    def _get_chrome_windows(self) -> list:
        """Get Chrome window information using system-specific methods."""
        windows = []
        
        if platform.system() == "Darwin":  # macOS
            try:
                # Use AppleScript to get Chrome window bounds
                script = '''
                tell application "System Events"
                    tell process "Google Chrome"
                        set windowList to every window
                        repeat with win in windowList
                            set winBounds to position of win & size of win
                            set winTitle to name of win
                        end repeat
                    end tell
                end tell
                '''
                # This is a simplified version - you might need to use pyobjc
                pass
            except:
                pass
        
        elif platform.system() == "Windows":
            try:
                import pygetwindow as gw
                chrome_windows = gw.getWindowsWithTitle("Chrome")
                for win in chrome_windows:
                    windows.append({
                        'left': win.left,
                        'top': win.top,
                        'width': win.width,
                        'height': win.height,
                        'title': win.title
                    })
            except ImportError:
                if self.debug:
                    print("pygetwindow not available for window detection")
        
        return windows

    def _detect_game_by_color_analysis(self, search_region: Optional[Dict] = None) -> Optional[Dict]:
        """
        Detect the dino game area by looking for characteristic colors and patterns.
        The dino game has a distinctive light gray background (#f7f7f7).
        """
        if search_region is None:
            # Search entire screen
            monitor = self.sct.monitors[0]  # All monitors
        else:
            monitor = search_region
            
        screenshot = np.array(self.sct.grab(monitor))[:, :, :3]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(screenshot, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Look for the light gray background of the dino game
        # The dino background is very light gray, almost white
        light_gray_mask = cv2.inRange(gray, 240, 255)
        
        # Morphological operations to clean up the mask
        kernel = np.ones((10, 10), np.uint8)
        light_gray_mask = cv2.morphologyEx(light_gray_mask, cv2.MORPH_CLOSE, kernel)
        light_gray_mask = cv2.morphologyEx(light_gray_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(light_gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Look for rectangular regions that could be the game canvas
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h
            
            # Dino game canvas is typically wider than tall, with substantial area
            if (area > 50000 and  # Minimum area
                aspect_ratio > 1.2 and  # Wider than tall
                aspect_ratio < 4.0 and  # Not too wide
                w > 300 and h > 150):  # Minimum dimensions
                
                candidates.append({
                    'left': x + (monitor['left'] if 'left' in monitor else 0),
                    'top': y + (monitor['top'] if 'top' in monitor else 0),
                    'width': w,
                    'height': h,
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        if not candidates:
            return None
            
        # Return the largest candidate
        best_candidate = max(candidates, key=lambda x: x['area'])
        
        if self.debug:
            print(f"Found game region: {best_candidate}")
            
        return best_candidate

    def _detect_game_by_template_matching(self, search_region: Optional[Dict] = None) -> Optional[Dict]:
        """
        Detect game area by looking for distinctive patterns like the ground line,
        cactus shapes, or the dino itself.
        """
        if search_region is None:
            monitor = self.sct.monitors[0]
        else:
            monitor = search_region
            
        screenshot = np.array(self.sct.grab(monitor))[:, :, :3]
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Look for horizontal lines (ground)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horizontal_kernel)
        horizontal_lines = cv2.erode(horizontal_lines, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        # Find contours of horizontal lines
        contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ground_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 200 and h < 10:  # Long horizontal line
                ground_candidates.append((x, y, w, h))
        
        if ground_candidates:
            # Use the ground line to estimate game area
            # Typically the game area extends above and to the sides of the ground
            gx, gy, gw, gh = max(ground_candidates, key=lambda x: x[2])  # Longest line
            
            # Estimate game canvas around the ground line
            canvas_padding = 50
            game_region = {
                'left': max(0, gx - canvas_padding) + (monitor.get('left', 0)),
                'top': max(0, gy - 200) + (monitor.get('top', 0)),  # Game area above ground
                'width': min(gw + 2 * canvas_padding, monitor.get('width', 1920)),
                'height': min(300, monitor.get('height', 1080))  # Reasonable height
            }
            
            if self.debug:
                print(f"Found game region via template matching: {game_region}")
                
            return game_region
            
        return None

    def _detect_game_region(self) -> Optional[Dict]:
        """
        Main detection method that tries multiple approaches.
        """
        if self.detection_method == "window":
            # Try window-based detection first
            chrome_windows = self._get_chrome_windows()
            for window in chrome_windows:
                if "dino" in window.get('title', '').lower():
                    return window
                    
        elif self.detection_method == "color":
            return self._detect_game_by_color_analysis()
            
        elif self.detection_method == "template":
            return self._detect_game_by_template_matching()
            
        else:  # auto mode
            # Try multiple methods in order
            methods = [
                ("color", self._detect_game_by_color_analysis),
                ("template", self._detect_game_by_template_matching),
            ]
            
            for method_name, method_func in methods:
                if self.debug:
                    print(f"Trying detection method: {method_name}")
                result = method_func()
                if result:
                    if self.debug:
                        print(f"Success with method: {method_name}")
                    return result
                    
        return None

    def _interactive_region_selection(self) -> Dict:
        """
        Fallback: Let user manually select the game region by clicking.
        """
        print("Automatic detection failed. Please manually select the game region.")
        print("Instructions:")
        print("1. A screenshot will be displayed")
        print("2. Click the top-left corner of the game area")
        print("3. Click the bottom-right corner of the game area")
        print("4. Close the image window when done")
        
        # Take screenshot
        monitor = self.sct.monitors[0]
        screenshot = np.array(self.sct.grab(monitor))[:, :, :3]
        screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        
        # Simple click handler for region selection
        clicks = []
        
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                clicks.append((int(event.xdata), int(event.ydata)))
                print(f"Clicked at: ({int(event.xdata)}, {int(event.ydata)})")
                if len(clicks) == 2:
                    plt.close()
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(screenshot_rgb)
        ax.set_title("Click top-left, then bottom-right of game area")
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        if len(clicks) == 2:
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            return {
                'left': min(x1, x2),
                'top': min(y1, y2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1)
            }
        else:
            # Default fallback
            return {'left': 200, 'top': 200, 'width': 600, 'height': 300}

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Preprocess frame to observation."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.use_blur:
            gray = cv2.GaussianBlur(gray, (3, 3), 1.0)
        if self.use_hist_eq:
            gray = cv2.equalizeHist(gray)
        # Resize to square observation space
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized.astype(np.uint8), axis=-1)

    def _grab_region(self, region: Dict) -> np.ndarray:
        """Grab a screen region."""
        try:
            return np.array(self.sct.grab(region))[:, :, :3].astype(np.uint8)
        except Exception as e:
            if self.debug:
                print(f"Screen capture error: {e}")
            # Return black frame as fallback
            return np.zeros((region['height'], region['width'], 3), dtype=np.uint8)

    def _detect_game_over(self) -> bool:
        """Detect game over state using multiple heuristics."""
        if self.game_region is None:
            return False
            
        # Grab current frame
        frame = self._grab_region(self.game_region)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Method 1: OCR detection
        try:
            text = pytesseract.image_to_string(gray, config='--psm 8').lower()
            if 'game over' in text or 'restart' in text:
                return True
        except:
            pass
            
        # Method 2: Look for restart button (circular arrow icon)
        # The restart button appears as a circular pattern
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=50)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            if len(circles) > 0:
                return True
                
        # Method 3: Check for static screen (no movement)
        # This would require storing previous frames
        
        return False

    def reset(self, *, seed: Optional[int] = None, options=None):
        """Reset environment and detect game region."""
        if self.debug:
            print("Detecting game region...")
            
        # Detect game region
        self.game_region = self._detect_game_region()
        
        if self.game_region is None:
            if self.debug:
                print("Automatic detection failed, trying interactive selection...")
            self.game_region = self._interactive_region_selection()
            
        if self.debug:
            print(f"Using game region: {self.game_region}")
        
        # Start/restart game
        keyboard.press_and_release('space')
        time.sleep(1.0)
        
        self.game_over = False
        self.step_count = 0
        
        # Get initial observation
        frame = self._grab_region(self.game_region)
        obs = self._preprocess(frame)
        
        return obs, {}

    def step(self, action: int):
        """Take an action in the environment."""
        # Execute action
        if action == 1:  # jump
            keyboard.press_and_release('up')
        elif action == 2:  # duck
            keyboard.press_and_release('down')
        # action == 0 is no-op
        
        time.sleep(self.action_sleep)
        
        # Get observation
        frame = self._grab_region(self.game_region)
        obs = self._preprocess(frame)
        
        self.step_count += 1
        
        # Check if game over
        done = self._detect_game_over()
        
        # Calculate reward
        if self.reward_mode == "sparse":
            reward = 1.0 if not done else -10.0
        else:  # shaped
            reward = 0.1 if not done else -1.0
            
        return obs, reward, done, False, {}

    def render(self):
        """Render the current state."""
        if self.game_region is None:
            return
            
        frame = self._grab_region(self.game_region)
        cv2.imshow('Chrome Dino Game', frame)
        cv2.waitKey(1)

    def close(self):
        """Clean up resources."""
        cv2.destroyAllWindows()

    def debug_show_detection(self):
        """Show the detected regions for debugging."""
        if self.game_region is None:
            print("No game region detected yet. Call reset() first.")
            return
            
        # Show the detected game region
        frame = self._grab_region(self.game_region)
        obs = self._preprocess(frame)
        
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Detected Game Region\n{self.game_region}')
        axes[0].axis('off')
        
        axes[1].imshow(obs.squeeze(), cmap='gray')
        axes[1].set_title('Preprocessed Observation')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()