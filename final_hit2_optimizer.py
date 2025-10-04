import pandas as pd
import numpy as np
from collections import Counter
import random

class FinalHit2Optimizer:
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
        
    def analyze_recent_patterns(self, data, lookback=8):
        """Analyze very recent patterns for maximum hit probability"""
        recent = data.tail(lookback)
        
        # Count all numbers in recent draws
        all_numbers = []
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            all_numbers.extend(numbers)
        
        freq = Counter(all_numbers)
        
        # Get numbers that appeared 2+ times in recent draws
        hot_repeaters = [num for num, count in freq.items() if count >= 2]
        
        # Get all frequent numbers
        frequent = [num for num, count in freq.most_common(15)]
        
        return hot_repeaters, frequent
    
    def get_number_clusters(self, data, lookback=12):
        """Find number ranges that appear together frequently"""
        recent = data.tail(lookback)
        
        low_range = []  # 1-23 for PB, 1-14 for MB
        mid_range = []  # 24-46 for PB, 15-28 for MB  
        high_range = [] # 47-69 for PB, 29-41 for MB
        
        max_num = 69 if len(data.columns) > 6 else 41  # PB vs MB detection
        
        for _, row in recent.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            for num in numbers:
                if num <= max_num // 3:
                    low_range.append(num)
                elif num <= 2 * max_num // 3:
                    mid_range.append(num)
                else:
                    high_range.append(num)
        
        low_freq = Counter(low_range).most_common(5)
        mid_freq = Counter(mid_range).most_common(5)
        high_freq = Counter(high_range).most_common(5)
        
        return ([num for num, count in low_freq],
                [num for num, count in mid_freq], 
                [num for num, count in high_freq])
    
    def predict_optimized(self, game_type='pb', is_double_play=False):
        """Optimized prediction focusing on recent hot patterns"""
        data = self.pb_data if game_type == 'pb' else self.mb_data
        max_num = 69 if game_type == 'pb' else 41
        
        # Get recent patterns
        hot_repeaters, frequent = self.analyze_recent_patterns(data, 8)
        low_nums, mid_nums, high_nums = self.get_number_clusters(data, 12)
        
        selected = []
        
        # Strategy 1: Use hot repeaters first (highest probability)
        if hot_repeaters:
            selected.extend(hot_repeaters[:3])  # Take up to 3 hot repeaters
        
        # Strategy 2: Ensure range distribution (balanced approach)
        ranges = [low_nums, mid_nums, high_nums]
        for range_nums in ranges:
            if len(selected) < 5 and range_nums:
                for num in range_nums:
                    if num not in selected:
                        selected.append(num)
                        break
        
        # Strategy 3: Fill with most frequent recent numbers
        for num in frequent:
            if len(selected) < 5 and num not in selected:
                selected.append(num)
        
        # Strategy 4: Fill remaining slots
        while len(selected) < 5:
            num = random.randint(1, max_num)
            if num not in selected:
                selected.append(num)
        
        selected = selected[:5]
        
        # Apply constraints while preserving hot numbers
        selected = self.apply_smart_constraints(selected, game_type, is_double_play)
        
        # Smart bonus selection
        bonus = self.predict_bonus(data, game_type)
        
        return selected, bonus
    
    def predict_bonus(self, data, game_type):
        """Predict bonus number based on recent patterns"""
        recent_bonuses = []
        
        if game_type == 'pb':
            for _, row in data.tail(15).iterrows():
                recent_bonuses.append(row['PB'])
            max_bonus = 26
        else:
            for _, row in data.tail(15).iterrows():
                recent_bonuses.append(row['Megaball'])
            max_bonus = 6
        
        bonus_freq = Counter(recent_bonuses)
        
        # 60% chance to use recent frequent bonus, 40% random
        if random.random() < 0.6 and bonus_freq:
            return random.choice([b for b, count in bonus_freq.most_common(3)])
        else:
            return random.randint(1, max_bonus)
    
    def apply_smart_constraints(self, numbers, game_type, is_double_play):
        """Apply constraints while preserving hot numbers"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        # Ensure unique numbers
        numbers = list(set(numbers))
        while len(numbers) < 5:
            numbers.append(random.randint(1, max_num))
        numbers = numbers[:5]
        
        # Constraint 2: Include previous draw number (REQUIRED)
        if len(data) > 0:
            last_numbers = [data.iloc[-1]['1'], data.iloc[-1]['2'], data.iloc[-1]['3'], 
                           data.iloc[-1]['4'], data.iloc[-1]['5']]
            # Replace the least "hot" number with previous draw number
            numbers[-1] = random.choice(last_numbers)
        
        # Constraint 3: Avoid last 4 games (except the repeat number)
        recent_numbers = set()
        if len(data) >= 4:
            for i in range(4):
                row = data.iloc[-(i+1)]
                recent_numbers.update([row['1'], row['2'], row['3'], row['4'], row['5']])
        
        # Only replace if not the repeat number
        for i in range(len(numbers)-1):
            attempts = 0
            while numbers[i] in recent_numbers and attempts < 20:
                numbers[i] = random.randint(1, max_num)
                attempts += 1
        
        # Constraint 1: Even/odd balance (try to maintain hot numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        
        if even_count not in [2, 3]:
            # Adjust the least important number (last one)
            if even_count < 2:
                # Need more evens
                if numbers[-1] % 2 == 1:
                    numbers[-1] = numbers[-1] + 1 if numbers[-1] < max_num else numbers[-1] - 1
            elif even_count > 3:
                # Need fewer evens  
                if numbers[-1] % 2 == 0:
                    numbers[-1] = numbers[-1] + 1 if numbers[-1] < max_num else numbers[-1] - 1
        
        # Constraint 4: Sum adjustment (minimal impact on hot numbers)
        total = sum(numbers)
        if game_type == 'pb':
            target_min, target_max = 70, (285 if is_double_play else 299)
        else:
            target_min, target_max = 36, 177
        
        if total < target_min:
            numbers[-1] += min(15, target_min - total)
        elif total > target_max:
            numbers[-1] -= min(15, total - target_max)
        
        numbers[-1] = max(1, min(max_num, numbers[-1]))
        
        return sorted(numbers)

def comprehensive_test():
    """Comprehensive test of the optimizer"""
    predictor = FinalHit2Optimizer()
    predictor.load_data()
    
    print("COMPREHENSIVE HIT-2 OPTIMIZER TEST")
    print("=" * 50)
    
    # Test different lookback periods to find best
    best_pb_rate = 0
    best_mb_rate = 0
    
    print("\nTesting Powerball (Last 25 draws):")
    pb_hits = []
    
    for i in range(25):
        test_idx = len(predictor.pb_data) - 25 + i
        actual_row = predictor.pb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.pb_data = predictor.pb_data.iloc[:test_idx]
        
        pred_numbers, pred_pb = predictor.predict_optimized('pb')
        hits = len(set(actual) & set(pred_numbers))
        pb_hits.append(hits)
        
        if hits >= 2:
            print(f"HIT! Test {i+1}: {hits} hits - Pred {pred_numbers} | Actual {actual}")
        elif hits == 1:
            print(f"Close Test {i+1}: {hits} hit - Pred {pred_numbers} | Actual {actual}")
    
    predictor.load_data()
    
    print("\nTesting Megabucks (Last 25 draws):")
    mb_hits = []
    
    for i in range(25):
        test_idx = len(predictor.mb_data) - 25 + i
        actual_row = predictor.mb_data.iloc[test_idx]
        actual = [actual_row['1'], actual_row['2'], actual_row['3'], actual_row['4'], actual_row['5']]
        
        predictor.mb_data = predictor.mb_data.iloc[:test_idx]
        
        pred_numbers, pred_mb = predictor.predict_optimized('mb')
        hits = len(set(actual) & set(pred_numbers))
        mb_hits.append(hits)
        
        if hits >= 2:
            print(f"HIT! Test {i+1}: {hits} hits - Pred {pred_numbers} | Actual {actual}")
        elif hits == 1:
            print(f"Close Test {i+1}: {hits} hit - Pred {pred_numbers} | Actual {actual}")
    
    # Final results
    pb_2plus = sum(1 for x in pb_hits if x >= 2)
    mb_2plus = sum(1 for x in mb_hits if x >= 2)
    total_2plus = pb_2plus + mb_2plus
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Powerball: {pb_2plus}/25 hit 2+ numbers ({pb_2plus/25*100:.1f}%)")
    print(f"Megabucks: {mb_2plus}/25 hit 2+ numbers ({mb_2plus/25*100:.1f}%)")
    print(f"OVERALL: {total_2plus}/50 hit 2+ numbers ({total_2plus/50*100:.1f}%)")
    print(f"Average PB hits: {np.mean(pb_hits):.2f}")
    print(f"Average MB hits: {np.mean(mb_hits):.2f}")
    
    if total_2plus >= 10:  # 20% success rate
        print("\nSUCCESS! Strategy achieving good hit rate!")
        return True
    else:
        print(f"\nNeed improvement. Target: 10+ hits, Achieved: {total_2plus}")
        return False

def main():
    predictor = FinalHit2Optimizer()
    predictor.load_data()
    
    # Run comprehensive test
    success = comprehensive_test()
    
    # Generate current predictions
    print("\n" + "=" * 50)
    print("OPTIMIZED PREDICTIONS FOR TODAY:")
    print("=" * 50)
    
    pb_pred = predictor.predict_optimized('pb', False)
    pb_dp_pred = predictor.predict_optimized('pb', True)
    mb_pred = predictor.predict_optimized('mb')
    
    print(f"\nPowerball Regular: {pb_pred[0]} + {pb_pred[1]}")
    print(f"Powerball Double Play: {pb_dp_pred[0]} + {pb_dp_pred[1]}")
    print(f"Megabucks: {mb_pred[0]} + {mb_pred[1]}")
    
    # Show analysis
    print("\nConstraint Check:")
    for name, pred in [("PB", pb_pred), ("PB-DP", pb_dp_pred), ("MB", mb_pred)]:
        numbers, bonus = pred
        even_count = sum(1 for n in numbers if n % 2 == 0)
        total = sum(numbers)
        print(f"{name}: Sum={total}, Even/Odd={even_count}E/{5-even_count}O")
    
    if success:
        print("\nThese predictions use the optimized strategy that showed good results!")
    else:
        print("\nStrategy is still being refined, but these are the best predictions available.")

if __name__ == "__main__":
    main()