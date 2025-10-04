import pandas as pd
import numpy as np
from fast_lottery_predictor import FastLotteryPredictor

def test_historical_accuracy():
    """Test predictions against recent historical draws"""
    print("Testing prediction accuracy against historical data...")
    print("=" * 60)
    
    predictor = FastLotteryPredictor()
    if not predictor.load_data():
        print("Failed to load data!")
        return
    
    # Test Powerball
    print("\nTesting POWERBALL predictions:")
    print("-" * 40)
    
    pb_results = []
    for i in range(5):  # Test last 5 draws
        test_idx = len(predictor.pb_data) - 5 + i
        actual_row = predictor.pb_data.iloc[test_idx]
        actual_numbers = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        actual_pb = actual_row['PB']
        
        # Use data up to test point for prediction
        predictor.pb_data = predictor.pb_data.iloc[:test_idx]
        
        # Make prediction
        pred_result = predictor.predict_fast('pb', False)
        if pred_result:
            pred_numbers, pred_pb = pred_result
            
            # Count matches
            matches = len(set(actual_numbers) & set(pred_numbers))
            pb_match = 1 if actual_pb == pred_pb else 0
            total_hits = matches + pb_match
            
            pb_results.append(total_hits)
            
            print(f"Draw {i+1}:")
            print(f"  Actual: {actual_numbers} + {actual_pb}")
            print(f"  Predicted: {pred_numbers} + {pred_pb}")
            print(f"  Hits: {matches} numbers + {pb_match} PB = {total_hits} total")
    
    # Restore full data
    predictor.load_data()
    
    # Test Megabucks
    print("\nTesting MEGABUCKS predictions:")
    print("-" * 40)
    
    mb_results = []
    for i in range(5):  # Test last 5 draws
        test_idx = len(predictor.mb_data) - 5 + i
        actual_row = predictor.mb_data.iloc[test_idx]
        actual_numbers = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        actual_mb = actual_row['Megaball']
        
        # Use data up to test point for prediction
        predictor.mb_data = predictor.mb_data.iloc[:test_idx]
        
        # Make prediction
        pred_result = predictor.predict_fast('mb')
        if pred_result:
            pred_numbers, pred_mb = pred_result
            
            # Count matches
            matches = len(set(actual_numbers) & set(pred_numbers))
            mb_match = 1 if actual_mb == pred_mb else 0
            total_hits = matches + mb_match
            
            mb_results.append(total_hits)
            
            print(f"Draw {i+1}:")
            print(f"  Actual: {actual_numbers} + {actual_mb}")
            print(f"  Predicted: {pred_numbers} + {pred_mb}")
            print(f"  Hits: {matches} numbers + {mb_match} MB = {total_hits} total")
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if pb_results:
        print(f"Powerball Results:")
        print(f"  Average hits: {np.mean(pb_results):.2f}")
        print(f"  Best performance: {max(pb_results)} hits")
        print(f"  Tests with 2+ hits: {sum(1 for x in pb_results if x >= 2)}/5")
        print(f"  Tests with 3+ hits: {sum(1 for x in pb_results if x >= 3)}/5")
    
    if mb_results:
        print(f"\nMegabucks Results:")
        print(f"  Average hits: {np.mean(mb_results):.2f}")
        print(f"  Best performance: {max(mb_results)} hits")
        print(f"  Tests with 2+ hits: {sum(1 for x in mb_results if x >= 2)}/5")
        print(f"  Tests with 3+ hits: {sum(1 for x in mb_results if x >= 3)}/5")

def test_constraint_compliance():
    """Test that predictions meet all constraints"""
    print("\n" + "=" * 60)
    print("TESTING CONSTRAINT COMPLIANCE")
    print("=" * 60)
    
    predictor = FastLotteryPredictor()
    if not predictor.load_data():
        return
    
    compliance_stats = {
        'pb_even_odd': 0,
        'pb_sum_range': 0,
        'pb_dp_sum_range': 0,
        'mb_even_odd': 0,
        'mb_sum_range': 0
    }
    
    num_tests = 20
    
    for i in range(num_tests):
        # Test Powerball
        pb_pred = predictor.predict_fast('pb', False)
        if pb_pred:
            numbers, pb = pb_pred
            even_count = sum(1 for n in numbers if n % 2 == 0)
            total = sum(numbers)
            
            if even_count in [2, 3]:
                compliance_stats['pb_even_odd'] += 1
            if 70 <= total <= 299:
                compliance_stats['pb_sum_range'] += 1
        
        # Test Powerball Double Play
        pb_dp_pred = predictor.predict_fast('pb', True)
        if pb_dp_pred:
            numbers, pb = pb_dp_pred
            total = sum(numbers)
            
            if 70 <= total <= 285:
                compliance_stats['pb_dp_sum_range'] += 1
        
        # Test Megabucks
        mb_pred = predictor.predict_fast('mb')
        if mb_pred:
            numbers, mb = mb_pred
            even_count = sum(1 for n in numbers if n % 2 == 0)
            total = sum(numbers)
            
            if even_count in [2, 3]:
                compliance_stats['mb_even_odd'] += 1
            if 36 <= total <= 177:
                compliance_stats['mb_sum_range'] += 1
    
    print(f"Constraint compliance (out of {num_tests} tests):")
    print(f"Powerball Even/Odd Balance: {compliance_stats['pb_even_odd']}/{num_tests} ({100*compliance_stats['pb_even_odd']/num_tests:.1f}%)")
    print(f"Powerball Sum Range: {compliance_stats['pb_sum_range']}/{num_tests} ({100*compliance_stats['pb_sum_range']/num_tests:.1f}%)")
    print(f"Powerball DP Sum Range: {compliance_stats['pb_dp_sum_range']}/{num_tests} ({100*compliance_stats['pb_dp_sum_range']/num_tests:.1f}%)")
    print(f"Megabucks Even/Odd Balance: {compliance_stats['mb_even_odd']}/{num_tests} ({100*compliance_stats['mb_even_odd']/num_tests:.1f}%)")
    print(f"Megabucks Sum Range: {compliance_stats['mb_sum_range']}/{num_tests} ({100*compliance_stats['mb_sum_range']/num_tests:.1f}%)")

def main():
    print("QUICK LOTTERY PREDICTION TEST")
    print("=" * 60)
    
    # Test accuracy
    test_historical_accuracy()
    
    # Test constraints
    test_constraint_compliance()
    
    print("\n" + "=" * 60)
    print("Testing completed! The system shows how well predictions")
    print("match historical draws and comply with all constraints.")

if __name__ == "__main__":
    main()