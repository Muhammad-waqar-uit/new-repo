# Advanced Lottery Prediction System

An AI/ML-powered lottery prediction system that uses machine learning algorithms to predict lottery combinations while adhering to specific constraints and patterns.

## Features

### Core Constraints
1. **Even/Odd Balance**: Ensures 2 even/3 odd or 3 even/2 odd combinations (no all even/odd)
2. **Previous Draw Integration**: Includes 1 number from the previous drawing
3. **Recent Number Avoidance**: Excludes numbers from the 4 most recent consecutive games
4. **Sum Range Validation**: 
   - Powerball: 70-299
   - Powerball Double Play: 70-285
   - Megabucks: 36-177
5. **Day-of-Week Patterns**: Uses statistical analysis of day-specific number frequencies
6. **Separate Powerball Logic**: Even/odd constraints apply only to main numbers, not Powerball

### Advanced AI Features
- **Ensemble Machine Learning**: Multiple algorithms (Random Forest, Gradient Boosting, Neural Networks)
- **Hot/Cold Number Analysis**: Identifies frequently and infrequently drawn numbers
- **Overdue Number Prediction**: Predicts numbers due for appearance based on gap analysis
- **Pattern Recognition**: Analyzes consecutive numbers, sum distributions, and historical patterns
- **Smart Selection Strategies**: Balances multiple factors for optimal number selection

## Files

- `run_lottery_ai.py` - Main execution script
- `enhanced_predictor.py` - Advanced AI predictor with ensemble models
- `lottery_predictor.py` - Basic ML predictor
- `test_predictions.py` - Testing and validation script
- `pb_results.csv` - Powerball historical data
- `mb_results.csv` - Megabucks historical data

## Installation

1. Install Python 3.7 or higher
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start
```bash
python run_lottery_ai.py
```

### Testing Performance
```bash
python test_predictions.py
```

### Individual Predictors
```bash
# Basic predictor
python lottery_predictor.py

# Enhanced predictor
python enhanced_predictor.py
```

## Output

The system generates predictions for:
- **Powerball Regular**: 5 numbers (1-69) + Powerball (1-26)
- **Powerball Double Play**: 5 numbers (1-69) + Powerball (1-26)
- **Megabucks**: 5 numbers (1-41) + Megaball (1-6)

Each prediction includes:
- Complete number combination
- Sum total
- Even/odd distribution
- Constraint validation status

## Algorithm Details

### Basic Predictor
- Uses Random Forest, Gradient Boosting, and Neural Networks
- Features include number frequencies, day patterns, and recent trends
- Applies all constraints post-prediction

### Enhanced Predictor
- Ensemble voting regressors for each number position
- Advanced feature engineering with polynomial features
- Hot/cold number analysis with multiple time windows
- Gap-based overdue number prediction
- Smart selection balancing multiple strategies
- Pattern recognition for consecutive numbers and sum distributions

### Constraint Application
1. **Even/Odd Balance**: Intelligently swaps numbers to achieve 2E/3O or 3E/2O
2. **Previous Draw**: Incorporates one number from the most recent drawing
3. **Recent Avoidance**: Excludes numbers from last 4 consecutive games
4. **Sum Adjustment**: Strategically modifies numbers to meet sum requirements
5. **Range Validation**: Ensures all numbers are within valid ranges

## Testing Results

The system includes comprehensive testing against historical data:
- Validates predictions against actual lottery draws
- Measures hit rates (2+ and 3+ correct numbers)
- Compares basic vs enhanced predictor performance
- Analyzes constraint compliance rates

## Data Format

### Powerball Data (pb_results.csv)
```
DrawDate,1,2,3,4,5,PB,PP*
Monday 9/29/2025,1,3,27,60,65,16,5
```

### Megabucks Data (mb_results.csv)
```
Date,1,2,3,4,5,Megaball
9/29/2025,14,21,29,32,35,6
```

## Disclaimer

This system is for educational and entertainment purposes only. Lottery games are games of chance, and no prediction system can guarantee winning numbers. Past performance does not indicate future results.

## Technical Notes

- The system uses scikit-learn for machine learning algorithms
- Features are scaled and normalized for optimal model performance
- Ensemble methods combine multiple algorithms for better predictions
- Constraint satisfaction is applied post-prediction to ensure compliance
- Historical pattern analysis improves number selection strategies

## Future Enhancements

- Deep learning models (LSTM, Transformer)
- More sophisticated pattern recognition
- Real-time data integration
- Advanced statistical analysis
- Multi-game correlation analysis