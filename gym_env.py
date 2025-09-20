import cv2
import numpy as np
import time
import keyboard
import pytesseract
from mss import mss
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import PPO

import matplotlib.pyplot as plt
import os


class ChromeDinoEnv(Env):
    def __init__(self):
        super(ChromeDinoEnv, self).__init__()
        
        # Define action space: 0 = do nothing, 1 = jump, 2 = duck
        self.action_space = Discrete(3)
        
        # Define observation space to match _get_observation output
        self.observation_space = Box(low=0, high=255, shape=(83, 100, 1), dtype=np.uint8)
        
        # Screen capture setup
        self.sct = mss()
        self.game_location = {'top': 110, 'left': 50, 'width': 600, 'height': 500}
        self.done_location = {'top': 200, 'left': 600, 'width': 600, 'height': 70}
        
        # Initialize game state
        self.game_over = False
        self.step_count = 0

    def reset(self, **kwargs):
        # Reset the game
        keyboard.press_and_release('space')
        time.sleep(1)
        
        self.game_over = False
        self.step_count = 0
        return self._get_observation(), {}

    def step(self, action):
        # Perform action
        if action == 1:
            keyboard.press_and_release('up')
        elif action == 2:
            keyboard.press_and_release('down')
        
        time.sleep(0.1)
        
        observation = self._get_observation()
        self.game_over, _ = self._check_game_over()
        self.step_count += 1
        
        # Define reward and termination conditions
        reward = 1 if not self.game_over else -10
        done = self.game_over  
        
        return observation, reward, done, False, {}


    def _get_observation(self):
        raw = np.array(self.sct.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        # Return with channels last (height, width, channels)
        return np.expand_dims(resized, axis=-1)
    
    def _check_game_over(self):
        done_cap = np.array(self.sct.grab(self.done_location))
        text = pytesseract.image_to_string(done_cap).lower()
        done = "game over" in text
        return done, done_cap

    def render(self, mode='human'):
        observation = self._get_observation().squeeze()
        observation = (observation * 255).astype(np.uint8)
        cv2.imshow('Game', observation)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()