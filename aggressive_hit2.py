import pandas as pd
import numpy as np
from collections import Counter
import random

class AggressiveHit2:
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
        
    def get_super_hot_numbers(self, data, lookback=15):
        """Get extremely hot numbers from very recent draws"""
        recent = data.tail(lookback)
        freq = Counter()
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            freq.update(numbers)
        
        # Return numbers that appeared 2+ times in recent draws
        hot = [num for num, count in freq.items() if count >= 2]
        return sorted(hot, key=lambda x: freq[x], reverse=True)
    
    def get_repeating_patterns(self, data, lookback=30):
        """Find numbers that tend to repeat within short periods"""
        recent = data.tail(lookback)
        all_numbers = []
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            all_numbers.extend(numbers)
        
        # Count frequency and return most common
        freq = Counter(all_numbers)
        return [num for num, count in freq.most_common(15)]
    
    def predict_aggressive(self, game_type='pb', is_double_play=False):
        """Aggressive strategy: Use only the hottest numbers"""
        data = self.pb_data if game_type == 'pb' else self.mb_data
        max_num = 69 if game_type == 'pb' else 41
        
        # Get super hot numbers (appeared multiple times recently)
        super_hot = self.get_super_hot_numbers(data, 12)
        repeating = self.get_repeating_patterns(data, 25)
        
        selected = []
        
        # Strategy 1: Use ALL super hot numbers if available
        if len(super_hot) >= 3:
            selected.extend(super_hot[:4])  # Take up to 4 super hot
        
        # Strategy 2: Fill with most repeating numbers
        for num in repeating:
            if num not in selected and len(selected) < 5:
                selected.append(num)
        
        # Strategy 3: If still need numbers, use recent frequency
        if len(selected) < 5:
            recent_freq = Counter()
            for _, row in data.tail(20).iterrows():
                numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
                recent_freq.update(numbers)
            
            for num, count in recent_freq.most_common():
                if num not in selected and len(selected) < 5:
                    selected.append(num)
        
        # Ensure we have 5 numbers
        while len(selected) < 5:
            selected.append(random.randint(1, max_num))
        
        selected = list(set(selected))[:5]
        
        # Apply minimal constraints (prioritize hot numbers)
        selected = self.apply_minimal_constraints(selected, game_type, is_double_play)
        
        bonus = random.randint(1, 26 if game_type == 'pb' else 6)
        return selected, bonus
    
    def apply_minimal_constraints(self, numbers, game_type, is_double_play):
        """Apply constraints but preserve hot numbers as much as possible"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        # Ensure 5 unique numbers
        numbers = list(set(numbers))
        while len(numbers) < 5:
            numbers.append(random.randint(1, max_num))
        numbers = numbers[:5]
        
        # Include previous draw number (constraint 2)
        if len(data) > 0:
            last_numbers = [data.iloc[-1]['1'], data.iloc[-1]['2'], data.iloc[-1]['3'], 
                           data.iloc[-1]['4'], data.iloc[-1]['5']]
            # Replace least important number with previous draw number
            numbers[-1] = random.choice(last_numbers)
        
        # Minimal even/odd adjustment (only if severely imbalanced)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        if even_count == 0:  # All odd
            numbers[-1] = numbers[-1] + 1 if numbers[-1] < max_num else numbers[-1] - 1
        elif even_count == 5:  # All even
            numbers[-1] = numbers[-1] + 1 if numbers[-1] < max_num else numbers[-1] - 1
        
        # Basic sum adjustment
        total = sum(numbers)
        if game_type == 'pb':
            target_min, target_max = 70, (285 if is_double_play else 299)
        else:
            target_min, target_max = 36, 177
        
        if total < target_min:
            numbers[-1] += min(10, target_min - total)
        elif total > target_max:
            numbers[-1] -= min(10, total - target_max)
        
        numbers[-1] = max(1, min(max_num, numbers[-1]))
        
        return sorted(numbers)

def test_aggressive():
    """Test aggressive strategy"""
    predictor = AggressiveHit2()
    predictor.load_data()
    
    print("AGGRESSIVE HIT-2 STRATEGY TEST")
    print("=" * 40)
    
    # Test more recent draws (better chance of patterns)
    pb_hits = []
    print("\nPOWERBALL (Last 15 draws):")
    
    for i in range(15):
        test_idx = len(predictor.pb_data) - 15 + i
        actual_row = predictor.pb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        # Use data up to test point
        predictor.pb_data = predictor.pb_data.iloc[:test_idx]
        
        pred_numbers, pred_pb = predictor.predict_aggressive('pb')
        hits = len(set(actual) & set(pred_numbers))
        pb_hits.append(hits)
        
        status = "✓" if hits >= 2 else "✗"
        print(f"{status} Test {i+1}: Predicted {pred_numbers} | Actual {actual} | Hits: {hits}")
    
    # Reload and test Megabucks
    predictor.load_data()
    mb_hits = []
    print("\nMEGABUCKS (Last 15 draws):")
    
    for i in range(15):
        test_idx = len(predictor.mb_data) - 15 + i
        actual_row = predictor.mb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.mb_data = predictor.mb_data.iloc[:test_idx]
        
        pred_numbers, pred_mb = predictor.predict_aggressive('mb')
        hits = len(set(actual) & set(pred_numbers))
        mb_hits.append(hits)
        
        status = "✓" if hits >= 2 else "✗"
        print(f"{status} Test {i+1}: Predicted {pred_numbers} | Actual {actual} | Hits: {hits}")
    
    # Results
    print("\n" + "=" * 40)
    print("AGGRESSIVE STRATEGY RESULTS:")
    pb_2plus = sum(1 for x in pb_hits if x >= 2)
    mb_2plus = sum(1 for x in mb_hits if x >= 2)
    total_2plus = pb_2plus + mb_2plus
    
    print(f"Powerball: {pb_2plus}/15 hit 2+ ({pb_2plus/15*100:.1f}%)")
    print(f"Megabucks: {mb_2plus}/15 hit 2+ ({mb_2plus/15*100:.1f}%)")
    print(f"OVERALL: {total_2plus}/30 hit 2+ ({total_2plus/30*100:.1f}%)")
    print(f"Average PB hits: {np.mean(pb_hits):.2f}")
    print(f"Average MB hits: {np.mean(mb_hits):.2f}")
    
    if total_2plus >= 6:  # 20% success rate
        print("\n✓ STRATEGY WORKING! Good hit rate achieved.")
    else:
        print("\n✗ Need to adjust strategy further.")

def main():
    predictor = AggressiveHit2()
    predictor.load_data()
    
    # Test the strategy
    test_aggressive()
    
    # Generate current predictions
    print("\n" + "=" * 40)
    print("CURRENT AGGRESSIVE PREDICTIONS:")
    
    pb_pred = predictor.predict_aggressive('pb', False)
    pb_dp_pred = predictor.predict_aggressive('pb', True)
    mb_pred = predictor.predict_aggressive('mb')
    
    print(f"\nPowerball: {pb_pred[0]} + {pb_pred[1]}")
    print(f"PB Double Play: {pb_dp_pred[0]} + {pb_dp_pred[1]}")
    print(f"Megabucks: {mb_pred[0]} + {mb_pred[1]}")
    
    print("\nStrategy: Using only the hottest repeating numbers!")

if __name__ == "__main__":
    main()