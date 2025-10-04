# Lottery Prediction System - Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Fork this repository** or create your own GitHub repository
2. **Go to [Streamlit Community Cloud](https://share.streamlit.io/)**
3. **Sign in with GitHub**
4. **Click "New app"**
5. **Configure**:
   - Repository: `your-username/lottery-prediction-system`
   - Branch: `main`
   - Main file path: `lottery_app.py`
   - Requirements file: `requirements_deploy.txt`

## Local Development

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/Scripts/activate  # Windows Git Bash
# or
venv\Scripts\activate  # Windows Command Prompt

# Install dependencies
pip install -r requirements_deploy.txt

# Run the app
streamlit run lottery_app.py
```

## Features

- 🎲 Powerball & Megabucks predictions
- 🤖 AI-powered number selection
- 📊 Hot/cold number analysis
- 🎯 Pattern recognition
- 📈 Interactive web interface

## Disclaimer

This system is for educational and entertainment purposes only. Lottery games are games of chance, and no prediction system can guarantee winning numbers.
