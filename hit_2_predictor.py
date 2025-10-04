import pandas as pd
import numpy as np
from collections import Counter
import random

class Hit2Predictor:
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
        
    def get_most_frequent_numbers(self, data, game_type, lookback=30):
        """Get most frequently drawn numbers in recent draws"""
        recent = data.tail(lookback)
        freq = Counter()
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            freq.update(numbers)
        
        # Return top frequent numbers
        return [num for num, count in freq.most_common(20)]
    
    def get_number_pairs(self, data, lookback=50):
        """Find numbers that often appear together"""
        recent = data.tail(lookback)
        pairs = Counter()
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    pairs[pair] += 1
        
        return pairs.most_common(10)
    
    def predict_hit2(self, game_type='pb', is_double_play=False):
        """Predict with focus on hitting 2+ numbers"""
        data = self.pb_data if game_type == 'pb' else self.mb_data
        max_num = 69 if game_type == 'pb' else 41
        
        # Strategy: Use most frequent recent numbers + smart selection
        hot_numbers = self.get_most_frequent_numbers(data, game_type, 25)
        pairs = self.get_number_pairs(data, 40)
        
        selected = []
        
        # 1. Pick 2-3 from hottest numbers (high probability)
        selected.extend(hot_numbers[:3])
        
        # 2. Add numbers from frequent pairs
        for (num1, num2), count in pairs[:3]:
            if num1 in selected and num2 not in selected:
                selected.append(num2)
                break
            elif num2 in selected and num1 not in selected:
                selected.append(num1)
                break
        
        # 3. Fill remaining with strategic selection
        while len(selected) < 5:
            # Pick from next tier of hot numbers
            for num in hot_numbers[3:15]:
                if num not in selected:
                    selected.append(num)
                    break
            else:
                selected.append(random.randint(1, max_num))
        
        selected = selected[:5]
        
        # Apply constraints
        selected = self.apply_constraints(selected, game_type, is_double_play)
        
        # Bonus number
        bonus = random.randint(1, 26 if game_type == 'pb' else 6)
        
        return selected, bonus
    
    def apply_constraints(self, numbers, game_type, is_double_play):
        """Apply constraints while preserving hot numbers"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        numbers = list(set(numbers))[:5]
        while len(numbers) < 5:
            numbers.append(random.randint(1, max_num))
        
        # Even/odd balance - try to keep hot numbers
        even_count = sum(1 for n in numbers if n % 2 == 0)
        target_even = 2 if even_count <= 2 else 3
        
        if even_count != target_even:
            # Adjust least important numbers first
            if even_count < target_even:
                for i in range(len(numbers)-1, -1, -1):
                    if numbers[i] % 2 == 1:
                        for replacement in range(2, max_num+1, 2):
                            if replacement not in numbers:
                                numbers[i] = replacement
                                break
                        break
            else:
                for i in range(len(numbers)-1, -1, -1):
                    if numbers[i] % 2 == 0:
                        for replacement in range(1, max_num+1, 2):
                            if replacement not in numbers:
                                numbers[i] = replacement
                                break
                        break
        
        # Include number from previous draw
        if len(data) > 0:
            last_numbers = [data.iloc[-1]['1'], data.iloc[-1]['2'], data.iloc[-1]['3'], 
                           data.iloc[-1]['4'], data.iloc[-1]['5']]
            numbers[0] = random.choice(last_numbers)
        
        # Avoid last 4 games (except position 0)
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
            numbers[-1] += (target_min - total)
        elif total > target_max:
            numbers[-1] -= (total - target_max)
        
        numbers[-1] = max(1, min(max_num, numbers[-1]))
        
        return sorted(numbers)

def test_hit_rate():
    """Test the predictor against historical data"""
    predictor = Hit2Predictor()
    predictor.load_data()
    
    print("Testing Hit Rate Against Historical Data")
    print("=" * 50)
    
    # Test Powerball
    pb_hits = []
    print("\nPOWERBALL TESTS:")
    
    for i in range(10):  # Test last 10 draws
        test_idx = len(predictor.pb_data) - 10 + i
        actual_row = predictor.pb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        # Use data up to test point
        predictor.pb_data = predictor.pb_data.iloc[:test_idx]
        
        pred_numbers, pred_pb = predictor.predict_hit2('pb')
        hits = len(set(actual) & set(pred_numbers))
        pb_hits.append(hits)
        
        print(f"Test {i+1}: Actual {actual} | Predicted {pred_numbers} | Hits: {hits}")
    
    # Reload full data
    predictor.load_data()
    
    # Test Megabucks
    mb_hits = []
    print("\nMEGABUCKS TESTS:")
    
    for i in range(10):
        test_idx = len(predictor.mb_data) - 10 + i
        actual_row = predictor.mb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.mb_data = predictor.mb_data.iloc[:test_idx]
        
        pred_numbers, pred_mb = predictor.predict_hit2('mb')
        hits = len(set(actual) & set(pred_numbers))
        mb_hits.append(hits)
        
        print(f"Test {i+1}: Actual {actual} | Predicted {pred_numbers} | Hits: {hits}")
    
    # Results
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY:")
    print(f"Powerball - Average hits: {np.mean(pb_hits):.2f}")
    print(f"Powerball - 2+ hits: {sum(1 for x in pb_hits if x >= 2)}/10")
    print(f"Powerball - Best: {max(pb_hits)} hits")
    
    print(f"Megabucks - Average hits: {np.mean(mb_hits):.2f}")
    print(f"Megabucks - 2+ hits: {sum(1 for x in mb_hits if x >= 2)}/10")
    print(f"Megabucks - Best: {max(mb_hits)} hits")
    
    total_2plus = sum(1 for x in pb_hits + mb_hits if x >= 2)
    print(f"\nOVERALL: {total_2plus}/20 tests hit 2+ numbers ({total_2plus/20*100:.1f}%)")

def main():
    predictor = Hit2Predictor()
    predictor.load_data()
    
    print("HIT-2 FOCUSED LOTTERY PREDICTOR")
    print("=" * 40)
    
    # Test first
    test_hit_rate()
    
    print("\n" + "=" * 40)
    print("CURRENT PREDICTIONS:")
    
    # Generate current predictions
    pb_pred = predictor.predict_hit2('pb', False)
    pb_dp_pred = predictor.predict_hit2('pb', True)
    mb_pred = predictor.predict_hit2('mb')
    
    print(f"\nPowerball: {pb_pred[0]} + {pb_pred[1]}")
    print(f"PB Double Play: {pb_dp_pred[0]} + {pb_dp_pred[1]}")
    print(f"Megabucks: {mb_pred[0]} + {mb_pred[1]}")

if __name__ == "__main__":
    main()