import pandas as pd
import numpy as np
from collections import Counter
import random

class StatisticalHit2:
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
        
    def get_most_drawn_numbers(self, data, lookback=100):
        """Get historically most drawn numbers"""
        recent = data.tail(lookback)
        freq = Counter()
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            freq.update(numbers)
        
        return [num for num, count in freq.most_common()]
    
    def get_number_pairs_that_hit(self, data, lookback=200):
        """Find pairs of numbers that often appear together"""
        recent = data.tail(lookback)
        pairs = Counter()
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    pairs[pair] += 1
        
        return pairs.most_common(20)
    
    def predict_statistical(self, game_type='pb', is_double_play=False):
        """Use pure statistical approach"""
        data = self.pb_data if game_type == 'pb' else self.mb_data
        max_num = 69 if game_type == 'pb' else 41
        
        # Get most frequent numbers over longer period
        most_drawn = self.get_most_drawn_numbers(data, 150)
        
        # Get successful pairs
        successful_pairs = self.get_number_pairs_that_hit(data, 300)
        
        selected = []
        
        # Strategy: Use top frequent numbers
        selected.extend(most_drawn[:5])
        
        # Try to include numbers from successful pairs
        for (num1, num2), count in successful_pairs[:10]:
            if num1 in selected and num2 not in selected and len(selected) < 5:
                selected.append(num2)
                break
            elif num2 in selected and num1 not in selected and len(selected) < 5:
                selected.append(num1)
                break
        
        selected = list(set(selected))[:5]
        
        # Fill if needed
        while len(selected) < 5:
            selected.append(most_drawn[len(selected)])
        
        # Apply constraints
        selected = self.apply_constraints(selected, game_type, is_double_play)
        
        bonus = random.randint(1, 26 if game_type == 'pb' else 6)
        return selected, bonus
    
    def apply_constraints(self, numbers, game_type, is_double_play):
        """Apply constraints"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        numbers = list(set(numbers))[:5]
        while len(numbers) < 5:
            numbers.append(random.randint(1, max_num))
        
        # Include previous draw
        if len(data) > 0:
            last_numbers = [data.iloc[-1]['1'], data.iloc[-1]['2'], data.iloc[-1]['3'], 
                           data.iloc[-1]['4'], data.iloc[-1]['5']]
            numbers[0] = random.choice(last_numbers)
        
        # Even/odd balance
        even_count = sum(1 for n in numbers if n % 2 == 0)
        if even_count not in [2, 3]:
            if even_count < 2:
                for i in range(len(numbers)):
                    if numbers[i] % 2 == 1:
                        numbers[i] = numbers[i] + 1 if numbers[i] < max_num else numbers[i] - 1
                        if sum(1 for n in numbers if n % 2 == 0) >= 2:
                            break
            elif even_count > 3:
                for i in range(len(numbers)):
                    if numbers[i] % 2 == 0:
                        numbers[i] = numbers[i] + 1 if numbers[i] < max_num else numbers[i] - 1
                        if sum(1 for n in numbers if n % 2 == 0) <= 3:
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

def test_statistical():
    """Test statistical approach"""
    predictor = StatisticalHit2()
    predictor.load_data()
    
    print("STATISTICAL APPROACH TEST")
    print("Testing against last 30 draws each")
    print("=" * 40)
    
    # Test Powerball
    pb_hits = []
    pb_2plus = 0
    
    print("\nPowerball Results:")
    for i in range(30):
        test_idx = len(predictor.pb_data) - 30 + i
        actual_row = predictor.pb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.pb_data = predictor.pb_data.iloc[:test_idx]
        
        pred_numbers, pred_pb = predictor.predict_statistical('pb')
        hits = len(set(actual) & set(pred_numbers))
        pb_hits.append(hits)
        
        if hits >= 2:
            pb_2plus += 1
            print(f"HIT {hits}! Test {i+1}: {pred_numbers} vs {actual}")
    
    # Test Megabucks
    predictor.load_data()
    mb_hits = []
    mb_2plus = 0
    
    print("\nMegabucks Results:")
    for i in range(30):
        test_idx = len(predictor.mb_data) - 30 + i
        actual_row = predictor.mb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.mb_data = predictor.mb_data.iloc[:test_idx]
        
        pred_numbers, pred_mb = predictor.predict_statistical('mb')
        hits = len(set(actual) & set(pred_numbers))
        mb_hits.append(hits)
        
        if hits >= 2:
            mb_2plus += 1
            print(f"HIT {hits}! Test {i+1}: {pred_numbers} vs {actual}")
    
    # Summary
    total_2plus = pb_2plus + mb_2plus
    print(f"\n" + "=" * 40)
    print("STATISTICAL RESULTS:")
    print(f"Powerball 2+ hits: {pb_2plus}/30 ({pb_2plus/30*100:.1f}%)")
    print(f"Megabucks 2+ hits: {mb_2plus}/30 ({mb_2plus/30*100:.1f}%)")
    print(f"TOTAL 2+ hits: {total_2plus}/60 ({total_2plus/60*100:.1f}%)")
    print(f"Average PB hits: {np.mean(pb_hits):.2f}")
    print(f"Average MB hits: {np.mean(mb_hits):.2f}")
    
    return total_2plus

def main():
    predictor = StatisticalHit2()
    predictor.load_data()
    
    # Test the approach
    total_hits = test_statistical()
    
    # Generate predictions
    print("\n" + "=" * 40)
    print("CURRENT STATISTICAL PREDICTIONS:")
    
    pb_pred = predictor.predict_statistical('pb', False)
    pb_dp_pred = predictor.predict_statistical('pb', True)
    mb_pred = predictor.predict_statistical('mb')
    
    print(f"\nPowerball Regular: {pb_pred[0]} + {pb_pred[1]}")
    print(f"Powerball Double Play: {pb_dp_pred[0]} + {pb_dp_pred[1]}")
    print(f"Megabucks: {mb_pred[0]} + {mb_pred[1]}")
    
    # Show top frequent numbers for reference
    print("\nMost frequent numbers (last 150 draws):")
    pb_freq = predictor.get_most_drawn_numbers(predictor.pb_data, 150)
    mb_freq = predictor.get_most_drawn_numbers(predictor.mb_data, 150)
    
    print(f"Powerball top 10: {pb_freq[:10]}")
    print(f"Megabucks top 10: {mb_freq[:10]}")
    
    if total_hits >= 12:  # 20% success rate
        print(f"\nGOOD RESULTS! {total_hits} total 2+ hits achieved.")
    else:
        print(f"\nResults: {total_hits} total 2+ hits. Still working on optimization.")

if __name__ == "__main__":
    main()