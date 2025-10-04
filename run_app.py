#!/usr/bin/env python3
"""
Lottery Prediction App Launcher
===============================
Simple launcher for the Streamlit lottery prediction app
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_app.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def run_app():
    """Run the Streamlit app"""
    try:
        print("🚀 Starting Lottery Prediction App...")
        print("📱 The app will open in your web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n" + "="*50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "lottery_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    print("🎰 AI LOTTERY PREDICTOR")
    print("=" * 30)
    
    # Check if requirements are installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("📦 Installing requirements...")
        if not install_requirements():
            print("❌ Please install requirements manually:")
            print("pip install -r requirements_app.txt")
            return
    
    # Check if data files exist
    if not os.path.exists('pb_results.csv') or not os.path.exists('mb_results.csv'):
        print("❌ Data files missing! Please ensure pb_results.csv and mb_results.csv are present.")
        return
    
    # Run the app
    run_app()

if __name__ == "__main__":
    main()