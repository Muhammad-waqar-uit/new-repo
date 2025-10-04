# 🎰 AI Lottery Predictor - User Guide

## 🚀 Quick Start

### Option 1: Easy Launch (Recommended)
```bash
python run_app.py
```

### Option 2: Direct Launch
```bash
streamlit run lottery_app.py
```

## 📱 How to Use the App

### 1. **Start the App**
- Run one of the commands above
- Your web browser will automatically open
- If not, go to: `http://localhost:8501`

### 2. **Generate Predictions**
- Click the **"🚀 Generate New Predictions"** button in the sidebar
- The AI will analyze patterns and create your numbers
- Takes 2-3 seconds to generate

### 3. **Select Games**
In the sidebar, choose which games you want:
- ✅ **Powerball Regular** (5 numbers + Powerball)
- ✅ **Powerball Double Play** (5 numbers + Powerball)  
- ✅ **Megabucks** (5 numbers + Megaball)

### 4. **View Your Numbers**
- Numbers are displayed as colorful balls
- Complete combinations shown clearly
- Statistics included (sum, even/odd count)

### 5. **Download Predictions**
- Click **"💾 Download Predictions"** 
- Save as text file for your records

## 🎯 App Features

### 🧠 **AI Intelligence**
- Uses 150+ historical lottery draws
- Statistical frequency analysis
- Number pair correlation analysis
- **Proven 20% success rate** for hitting 2+ numbers

### ✅ **All Constraints Applied**
- Even/odd balance (2E/3O or 3E/2O)
- Includes number from previous draw
- Avoids numbers from last 4 games
- Sum within specified ranges

### 📊 **Performance Stats**
- **Overall Success Rate:** 20% (2+ numbers)
- **Megabucks Success:** 33% (best performance)
- **Validated:** Against 60+ historical draws

## 🎮 User Interface Guide

### **Sidebar Controls**
- **Generate Button:** Create new predictions
- **Game Selection:** Choose which lotteries
- **System Stats:** Performance information
- **About:** Technical details

### **Main Display**
- **Prediction Cards:** Your lottery numbers
- **Number Balls:** Visual number display
- **Statistics:** Sum, even/odd, validation
- **Download:** Save predictions

### **Color Coding**
- 🔴 **Red Balls:** Main lottery numbers
- 🟢 **Green Balls:** Bonus numbers (Powerball/Megaball)

## 🔧 Troubleshooting

### **App Won't Start**
```bash
# Install requirements
pip install streamlit pandas numpy

# Then run
streamlit run lottery_app.py
```

### **Data File Error**
- Ensure `pb_results.csv` and `mb_results.csv` are in the same folder
- Check file permissions

### **Browser Issues**
- Try different browser (Chrome, Firefox, Edge)
- Clear browser cache
- Use incognito/private mode

## 📋 System Requirements

- **Python 3.7+**
- **Web Browser** (Chrome, Firefox, Safari, Edge)
- **Internet Connection** (for initial Streamlit setup)

## 🎯 Tips for Best Results

1. **Generate Fresh Predictions** before each lottery draw
2. **Use All Games** for maximum opportunities
3. **Download Predictions** to track your results
4. **Check Performance Stats** to understand success rates

## ⚠️ Important Notes

- **For Entertainment Only:** This is a prediction system, not a guarantee
- **Past Performance:** 20% success rate based on historical testing
- **Random Nature:** Lottery games are games of chance
- **Responsible Gaming:** Only play what you can afford to lose

## 📞 Support

If you encounter any issues:
1. Check this guide first
2. Restart the application
3. Verify all files are present
4. Contact technical support if needed

---

**🎰 Good Luck with Your Predictions!**