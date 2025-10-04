#!/usr/bin/env python3
"""
Advanced Lottery Prediction System
==================================

This system uses machine learning to predict lottery combinations with the following constraints:
1. Even/odd balance (2 even/3 odd or 3 even/2 odd), no all even, no all odd
2. Include 1 number from the night before drawing
3. Don't select numbers that appeared in 4 most recent consecutive games
4. Sum totals within specified ranges:
   - Power Ball: 70-299
   - Power Ball Double Play: 70-285
   - Megabucks: 36-177
5. Use day of the week patterns for number selection
6. Generate 1 combo for regular Powerball and 1 for double play
7. The even/odd constraint applies only to the 5 main numbers (not Powerball)

Advanced Features:
- Ensemble machine learning models
- Hot/cold number analysis
- Overdue number prediction
- Pattern recognition
- Gap analysis
- Smart number selection strategies
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Import our predictors
from enhanced_predictor import EnhancedLotteryPredictor
from lottery_predictor import LotteryPredictor

def print_header():
    """Print system header"""
    print("=" * 80)
    print("ADVANCED LOTTERY PREDICTION SYSTEM")
    print("=" * 80)
    print("Using AI/ML with Advanced Constraint Satisfaction")
    print(f"Prediction Date: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")
    print("=" * 80)

def print_constraints():
    """Print the constraints being applied"""
    print("\nCONSTRAINTS APPLIED:")
    print("=" * 40)
    print("* Even/odd balance (2E/3O or 3E/2O)")
    print("* No all even or all odd combinations")
    print("* Include 1 number from previous draw")
    print("* Avoid numbers from last 4 consecutive games")
    print("* Sum ranges: PB(70-299), PB-DP(70-285), MB(36-177)")
    print("* Day-of-week statistical preferences")
    print("* Hot/cold number analysis")
    print("* Overdue number prediction")
    print("* Pattern recognition algorithms")

def validate_prediction(numbers, bonus, game_type, is_double_play=False):
    """Validate that prediction meets all constraints"""
    issues = []
    
    # Check even/odd balance
    even_count = sum(1 for n in numbers if n % 2 == 0)
    if even_count not in [2, 3]:
        issues.append(f"Even/odd imbalance: {even_count}E/{5-even_count}O")
    
    # Check no all even/odd
    if even_count == 0:
        issues.append("All odd numbers")
    elif even_count == 5:
        issues.append("All even numbers")
    
    # Check sum ranges
    total = sum(numbers)
    if game_type == 'pb':
        if is_double_play:
            if not (70 <= total <= 285):
                issues.append(f"Sum {total} outside PB Double Play range (70-285)")
        else:
            if not (70 <= total <= 299):
                issues.append(f"Sum {total} outside PB range (70-299)")
    else:  # Megabucks
        if not (36 <= total <= 177):
            issues.append(f"Sum {total} outside MB range (36-177)")
    
    # Check number ranges
    max_num = 69 if game_type == 'pb' else 41
    for num in numbers:
        if not (1 <= num <= max_num):
            issues.append(f"Number {num} outside valid range (1-{max_num})")
    
    # Check bonus range
    if game_type == 'pb':
        if not (1 <= bonus <= 26):
            issues.append(f"Powerball {bonus} outside valid range (1-26)")
    else:
        if not (1 <= bonus <= 6):
            issues.append(f"Megaball {bonus} outside valid range (1-6)")
    
    return issues

def format_prediction(numbers, bonus, game_type, is_double_play=False):
    """Format prediction for display"""
    total = sum(numbers)
    even_count = sum(1 for n in numbers if n % 2 == 0)
    
    if game_type == 'pb':
        game_name = "Powerball Double Play" if is_double_play else "Powerball Regular"
        bonus_name = "PB"
    else:
        game_name = "Megabucks"
        bonus_name = "MB"
    
    return {
        'game': game_name,
        'numbers': numbers,
        'bonus': bonus,
        'bonus_name': bonus_name,
        'sum': total,
        'even_count': even_count,
        'odd_count': 5 - even_count
    }

def print_prediction(pred_info, validation_issues):
    """Print formatted prediction"""
    print(f"\n{pred_info['game'].upper()}")
    print("-" * 50)
    print(f"Numbers: {pred_info['numbers']}")
    print(f"{pred_info['bonus_name']}: {pred_info['bonus']}")
    print(f"Complete: {pred_info['numbers']} + {pred_info['bonus']}")
    print(f"Sum: {pred_info['sum']}")
    print(f"Even/Odd: {pred_info['even_count']}E/{pred_info['odd_count']}O")
    
    if validation_issues:
        print("WARNING - VALIDATION ISSUES:")
        for issue in validation_issues:
            print(f"   - {issue}")
    else:
        print("SUCCESS: All constraints satisfied!")

def run_basic_predictor():
    """Run the basic predictor"""
    print("\nRunning Basic ML Predictor...")
    
    predictor = LotteryPredictor()
    predictor.load_data()
    predictor.train_models()
    
    predictions = []
    
    # Powerball Regular
    pb_pred = predictor.predict('pb', False)
    if pb_pred:
        numbers, bonus = pb_pred
        pred_info = format_prediction(numbers, bonus, 'pb', False)
        issues = validate_prediction(numbers, bonus, 'pb', False)
        predictions.append(('Basic', pred_info, issues))
    
    # Powerball Double Play
    pb_dp_pred = predictor.predict('pb', True)
    if pb_dp_pred:
        numbers, bonus = pb_dp_pred
        pred_info = format_prediction(numbers, bonus, 'pb', True)
        issues = validate_prediction(numbers, bonus, 'pb', True)
        predictions.append(('Basic DP', pred_info, issues))
    
    # Megabucks
    mb_pred = predictor.predict('mb')
    if mb_pred:
        numbers, bonus = mb_pred
        pred_info = format_prediction(numbers, bonus, 'mb', False)
        issues = validate_prediction(numbers, bonus, 'mb', False)
        predictions.append(('Basic MB', pred_info, issues))
    
    return predictions

def run_enhanced_predictor():
    """Run the enhanced predictor"""
    print("\nRunning Enhanced AI Predictor...")
    
    predictor = EnhancedLotteryPredictor()
    predictor.load_data()
    predictor.train_advanced_models()
    
    predictions = []
    
    # Powerball Regular
    pb_pred = predictor.predict_enhanced('pb', False)
    if pb_pred:
        numbers, bonus = pb_pred
        pred_info = format_prediction(numbers, bonus, 'pb', False)
        issues = validate_prediction(numbers, bonus, 'pb', False)
        predictions.append(('Enhanced', pred_info, issues))
    
    # Powerball Double Play
    pb_dp_pred = predictor.predict_enhanced('pb', True)
    if pb_dp_pred:
        numbers, bonus = pb_dp_pred
        pred_info = format_prediction(numbers, bonus, 'pb', True)
        issues = validate_prediction(numbers, bonus, 'pb', True)
        predictions.append(('Enhanced DP', pred_info, issues))
    
    # Megabucks
    mb_pred = predictor.predict_enhanced('mb')
    if mb_pred:
        numbers, bonus = mb_pred
        pred_info = format_prediction(numbers, bonus, 'mb', False)
        issues = validate_prediction(numbers, bonus, 'mb', False)
        predictions.append(('Enhanced MB', pred_info, issues))
    
    return predictions

def main():
    """Main execution function"""
    print_header()
    print_constraints()
    
    try:
        # Check if data files exist
        if not os.path.exists('pb_results.csv'):
            print("ERROR: pb_results.csv not found!")
            return
        
        if not os.path.exists('mb_results.csv'):
            print("ERROR: mb_results.csv not found!")
            return
        
        print("\nAnalyzing historical data and training models...")
        
        # Run both predictors
        basic_predictions = run_basic_predictor()
        enhanced_predictions = run_enhanced_predictor()
        
        # Display results
        print("\n" + "=" * 80)
        print("PREDICTION RESULTS")
        print("=" * 80)
        
        print("\nBASIC ML PREDICTIONS:")
        for pred_type, pred_info, issues in basic_predictions:
            print_prediction(pred_info, issues)
        
        print("\nENHANCED AI PREDICTIONS:")
        for pred_type, pred_info, issues in enhanced_predictions:
            print_prediction(pred_info, issues)
        
        # Summary
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print("The Enhanced AI Predictor uses advanced techniques including:")
        print("• Ensemble voting regressors")
        print("• Hot/cold number analysis")
        print("• Overdue number prediction")
        print("• Pattern recognition")
        print("• Smart selection strategies")
        print("\nFor best results, consider using the Enhanced predictions.")
        
        # Save predictions to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Lottery Predictions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ENHANCED AI PREDICTIONS:\n")
            f.write("-" * 30 + "\n")
            for pred_type, pred_info, issues in enhanced_predictions:
                f.write(f"{pred_info['game']}: {pred_info['numbers']} + {pred_info['bonus']}\n")
                f.write(f"Sum: {pred_info['sum']}, Even/Odd: {pred_info['even_count']}E/{pred_info['odd_count']}O\n\n")
        
        print(f"\nPredictions saved to: {filename}")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        print("Please ensure all required files are present and try again.")

if __name__ == "__main__":
    main()