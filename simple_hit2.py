import pandas as pd
import numpy as np
from collections import Counter
import random

class SimpleHit2:
    def __init__(self):
        self.pb_data = None
        self.mb_data = None
        
    def load_data(self):
        self.pb_data = pd.read_csv('pb_results.csv')
        self.pb_data['DrawDate'] = pd.to_datetime(self.pb_data['DrawDate'].str.split(' - ').str[0], errors='coerce')
        self.pb_data = self.pb_data.dropna().sort_values('DrawDate')
        
        self.mb_data = pd.read_csv('mb_results.csv')
        self.mb_data['Date'] = pd.to_datetime(self.mb_data['Date'], errors='coerce')
        self.mb_data = self.mb_data.dropna().sort_values('Date')
        
    def get_hot_numbers(self, data, lookback=10):
        """Get hottest numbers from very recent draws"""
        recent = data.tail(lookback)
        freq = Counter()
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            freq.update(numbers)
        
        return [num for num, count in freq.most_common(20)]
    
    def predict_simple(self, game_type='pb', is_double_play=False):
        """Simple strategy: Use hottest recent numbers + constraints"""
        data = self.pb_data if game_type == 'pb' else self.mb_data
        max_num = 69 if game_type == 'pb' else 41
        
        # Get hot numbers from last 10 draws
        hot_numbers = self.get_hot_numbers(data, 10)
        
        # Take top 5 hot numbers
        selected = hot_numbers[:5] if len(hot_numbers) >= 5 else hot_numbers
        
        # Fill if needed
        while len(selected) < 5:
            num = random.randint(1, max_num)
            if num not in selected:
                selected.append(num)
        
        # Apply constraints
        selected = self.apply_constraints(selected, game_type, is_double_play)
        
        bonus = random.randint(1, 26 if game_type == 'pb' else 6)
        return selected, bonus
    
    def apply_constraints(self, numbers, game_type, is_double_play):
        """Apply all constraints"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        numbers = list(set(numbers))[:5]
        while len(numbers) < 5:
            numbers.append(random.randint(1, max_num))
        
        # Even/odd balance
        even_count = sum(1 for n in numbers if n % 2 == 0)
        if even_count not in [2, 3]:
            if even_count < 2:
                # Need more evens
                for i in range(len(numbers)):
                    if numbers[i] % 2 == 1:
                        numbers[i] = numbers[i] + 1 if numbers[i] < max_num else numbers[i] - 1
                        if sum(1 for n in numbers if n % 2 == 0) >= 2:
                            break
            elif even_count > 3:
                # Need fewer evens
                for i in range(len(numbers)):
                    if numbers[i] % 2 == 0:
                        numbers[i] = numbers[i] + 1 if numbers[i] < max_num else numbers[i] - 1
                        if sum(1 for n in numbers if n % 2 == 0) <= 3:
                            break
        
        # Include previous draw number
        if len(data) > 0:
            last_numbers = [data.iloc[-1]['1'], data.iloc[-1]['2'], data.iloc[-1]['3'], 
                           data.iloc[-1]['4'], data.iloc[-1]['5']]
            numbers[0] = random.choice(last_numbers)
        
        # Avoid last 4 games
        recent_numbers = set()
        if len(data) >= 4:
            for i in range(4):
                row = data.iloc[-(i+1)]
                recent_numbers.update([row['1'], row['2'], row['3'], row['4'], row['5']])
        
        for i in range(1, len(numbers)):
            if numbers[i] in recent_numbers:
                for replacement in range(1, max_num+1):
                    if replacement not in numbers and replacement not in recent_numbers:
                        numbers[i] = replacement
                        break
        
        # Sum adjustment
        total = sum(numbers)
        if game_type == 'pb':
            target_min, target_max = 70, (285 if is_double_play else 299)
        else:
            target_min, target_max = 36, 177
        
        if total < target_min:
            diff = target_min - total
            numbers[-1] += diff
        elif total > target_max:
            diff = total - target_max
            numbers[-1] -= diff
        
        numbers[-1] = max(1, min(max_num, numbers[-1]))
        
        return sorted(numbers)

def test_simple():
    """Test simple strategy against recent draws"""
    predictor = SimpleHit2()
    predictor.load_data()
    
    print("SIMPLE HIT-2 STRATEGY TEST")
    print("=" * 40)
    
    # Test Powerball
    pb_hits = []
    print("\nPOWERBALL TESTS (Last 20 draws):")
    
    for i in range(20):
        test_idx = len(predictor.pb_data) - 20 + i
        actual_row = predictor.pb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        # Use data up to test point
        predictor.pb_data = predictor.pb_data.iloc[:test_idx]
        
        pred_numbers, pred_pb = predictor.predict_simple('pb')
        hits = len(set(actual) & set(pred_numbers))
        pb_hits.append(hits)
        
        status = "HIT" if hits >= 2 else "miss"
        print(f"{status} - Test {i+1}: Pred {pred_numbers} | Actual {actual} | Hits: {hits}")
    
    # Reload and test Megabucks
    predictor.load_data()
    mb_hits = []
    print("\nMEGABUCKS TESTS (Last 20 draws):")
    
    for i in range(20):
        test_idx = len(predictor.mb_data) - 20 + i
        actual_row = predictor.mb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.mb_data = predictor.mb_data.iloc[:test_idx]
        
        pred_numbers, pred_mb = predictor.predict_simple('mb')
        hits = len(set(actual) & set(pred_numbers))
        mb_hits.append(hits)
        
        status = "HIT" if hits >= 2 else "miss"
        print(f"{status} - Test {i+1}: Pred {pred_numbers} | Actual {actual} | Hits: {hits}")
    
    # Results
    print("\n" + "=" * 40)
    print("RESULTS:")
    pb_2plus = sum(1 for x in pb_hits if x >= 2)
    mb_2plus = sum(1 for x in mb_hits if x >= 2)
    total_2plus = pb_2plus + mb_2plus
    
    print(f"Powerball 2+ hits: {pb_2plus}/20 ({pb_2plus/20*100:.1f}%)")
    print(f"Megabucks 2+ hits: {mb_2plus}/20 ({mb_2plus/20*100:.1f}%)")
    print(f"TOTAL 2+ hits: {total_2plus}/40 ({total_2plus/40*100:.1f}%)")
    print(f"Average PB hits: {np.mean(pb_hits):.2f}")
    print(f"Average MB hits: {np.mean(mb_hits):.2f}")
    
    return total_2plus >= 8  # 20% success rate

def main():
    predictor = SimpleHit2()
    predictor.load_data()
    
    # Test the strategy
    success = test_simple()
    
    if success:
        print("\nSTRATEGY IS WORKING! Generating current predictions...")
    else:
        print("\nStrategy needs improvement, but generating predictions anyway...")
    
    # Generate current predictions
    print("\n" + "=" * 40)
    print("CURRENT PREDICTIONS:")
    
    pb_pred = predictor.predict_simple('pb', False)
    pb_dp_pred = predictor.predict_simple('pb', True)
    mb_pred = predictor.predict_simple('mb')
    
    print(f"\nPowerball Regular: {pb_pred[0]} + {pb_pred[1]}")
    print(f"Powerball Double Play: {pb_dp_pred[0]} + {pb_dp_pred[1]}")
    print(f"Megabucks: {mb_pred[0]} + {mb_pred[1]}")
    
    # Show constraint compliance
    for name, pred in [("PB", pb_pred), ("PB-DP", pb_dp_pred), ("MB", mb_pred)]:
        numbers, bonus = pred
        even_count = sum(1 for n in numbers if n % 2 == 0)
        total = sum(numbers)
        print(f"{name}: Sum={total}, Even/Odd={even_count}E/{5-even_count}O")

if __name__ == "__main__":
    main()