#!/usr/bin/env python3
"""
Test script for the dynamic Chrome Dino environment.
This script will automatically detect the game window and regions.
"""

import webbrowser
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from envs.dynamic_dino_env import DynamicChromeDinoEnv

def main():
    print("=== Dynamic Chrome Dino Environment Test ===\n")
    
    # Step 1: Open Chrome Dino game
    print("1. Opening Chrome Dino game...")
    webbrowser.open("chrome://dino")
    print("   Please wait for the game to load...")
    time.sleep(3)
    
    print("   Make sure the Chrome Dino game is visible on your screen!")
    print("   The T-Rex should be displayed and ready to play.")
    input("   Press Enter when the game is visible and ready...")
    
    # Step 2: Initialize environment with different detection methods
    detection_methods = ["color", "template", "auto"]
    
    for method in detection_methods:
        print(f"\n2. Testing detection method: '{method}'")
        
        try:
            # Create environment
            env = DynamicChromeDinoEnv(
                detection_method=method,
                debug=True,
                action_sleep=0.1
            )
            
            # Reset environment (this triggers detection)
            print("   Attempting to detect game region...")
            obs, info = env.reset()
            
            if obs is not None:
                print(f"   ✓ Success! Game region detected with method '{method}'")
                print(f"   Observation shape: {obs.shape}")
                
                # Show detection results
                env.debug_show_detection()
                
                # Test a few actions
                print("   Testing game actions...")
                test_actions(env)
                
                env.close()
                break
                
            else:
                print(f"   ✗ Failed with method '{method}'")
                env.close()
                continue
                
        except Exception as e:
            print(f"   ✗ Error with method '{method}': {e}")
            continue
    
    else:
        print("\n❌ All automatic detection methods failed!")
        print("Falling back to manual selection...")
        
        # Manual selection fallback
        env = DynamicChromeDinoEnv(detection_method="auto", debug=True)
        obs, info = env.reset()  # This will trigger interactive selection
        
        if obs is not None:
            print("✓ Manual selection successful!")
            env.debug_show_detection()
            test_actions(env)
        
        env.close()

def test_actions(env, num_steps=10):
    """Test the environment with random actions."""
    print(f"   Running {num_steps} test steps...")
    
    total_reward = 0
    for step in range(num_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        action_names = ["no-op", "jump", "duck"]
        print(f"     Step {step+1}: {action_names[action]} -> reward={reward:.1f}")
        
        if done:
            print("     Game over detected! Resetting...")
            env.reset()
            break
        
        time.sleep(0.2)  # Slow down for visibility
    
    print(f"   Test completed. Total reward: {total_reward:.1f}")

def debug_screen_capture():
    """Debug function to test basic screen capture capabilities."""
    print("\n=== Screen Capture Debug ===")
    
    try:
        from mss import mss
        sct = mss()
        
        print(f"Available monitors: {len(sct.monitors)}")
        for i, monitor in enumerate(sct.monitors):
            print(f"  Monitor {i}: {monitor}")
        
        # Test capture
        screenshot = sct.grab(sct.monitors[0])
        print("✓ Screen capture working!")
        return True
        
    except Exception as e:
        print(f"✗ Screen capture failed: {e}")
        
        # Try alternative methods
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            print("✓ PyAutoGUI screen capture working!")
            return True
        except Exception as e2:
            print(f"✗ PyAutoGUI also failed: {e2}")
            return False

def check_dependencies():
    """Check if all required dependencies are available."""
    print("=== Dependency Check ===")
    
    required_packages = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("mss", "mss"),
        ("keyboard", "keyboard"),
        ("gymnasium", "gymnasium"),
        ("pytesseract", "pytesseract")
    ]
    
    missing = []
    for package, pip_name in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (install with: pip install {pip_name})")
            missing.append(pip_name)
    
    optional_packages = [
        ("pyautogui", "pyautogui"),
        ("matplotlib", "matplotlib")
    ]
    
    for package, pip_name in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"! {package} (optional - install with: pip install {pip_name})")
    
    if missing:
        print(f"\n❌ Missing required packages. Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("✓ All required dependencies found!")
    return True

if __name__ == "__main__":
    # Check dependencies first
    if not check_dependencies():
        exit(1)
    
    # Debug screen capture
    if not debug_screen_capture():
        print("\n❌ Screen capture not working. Check permissions!")
        print("On macOS: System Preferences > Security & Privacy > Privacy > Screen Recording")
        exit(1)
    
    # Run main test
    try:
        main()
        print("\n🎉 Test completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()