import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

class DinoGameBot:
    def __init__(self):
        self.driver = self._launch_browser()
        self.actions = ActionChains(self.driver)

    def _launch_browser(self):
        options = Options()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)
        time.sleep(2)  # Allow window to fully open
        return driver

    def open_dino_via_address_bar(self):
        print("Navigating to chrome://dino")
        pyautogui.hotkey("ctrl", "l")
        time.sleep(0.3)
        pyautogui.typewrite("chrome://dino", interval=0.05)
        pyautogui.press("enter")
        time.sleep(2)  # Let the Dino game load

    def wait_for_game_ready(self):
        print("Waiting for Dino game canvas to appear...")
        while True:
            try:
                canvas = self.driver.find_element(By.CLASS_NAME, "runner-canvas")
                if canvas.is_displayed():
                    print("Game is ready.")
                    break
            except:
                pass
            time.sleep(0.5)

    def start_game(self):
        print("Starting game...")
        self.actions.send_keys(" ").perform()
        time.sleep(1)

    def jump(self):
        print("Jump!")
        self.actions.send_keys(" ").perform()

    def run(self):
        self.open_dino_via_address_bar()
        self.wait_for_game_ready()
        self.start_game()
        try:
            while True:
                self.jump()
                time.sleep(2)
        except KeyboardInterrupt:
            print("Exiting...")
            self.driver.quit()


if __name__ == "__main__":
    bot = DinoGameBot()
    bot.run()