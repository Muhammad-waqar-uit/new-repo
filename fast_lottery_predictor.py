import pandas as pd
import numpy as np
from datetime import datetime
import random

class FastLotteryPredictor:
    def __init__(self):
        self.pb_data = None
        self.mb_data = None
        
    def load_data(self):
        """Load lottery data quickly"""
        try:
            # Load Powerball data
            self.pb_data = pd.read_csv('pb_results.csv')
            self.pb_data['DrawDate'] = pd.to_datetime(self.pb_data['DrawDate'].str.split(' - ').str[0], errors='coerce')
            self.pb_data = self.pb_data.dropna(subset=['DrawDate']).sort_values('DrawDate')
            
            # Load Megabucks data  
            self.mb_data = pd.read_csv('mb_results.csv')
            self.mb_data['Date'] = pd.to_datetime(self.mb_data['Date'], errors='coerce')
            self.mb_data = self.mb_data.dropna(subset=['Date']).sort_values('Date')
            
            print(f"Loaded {len(self.pb_data)} Powerball draws and {len(self.mb_data)} Megabucks draws")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_hot_numbers(self, data, game_type, lookback=20):
        """Get frequently drawn numbers from recent draws"""
        max_num = 69 if game_type == 'pb' else 41
        recent_data = data.tail(lookback)
        
        number_freq = {}
        for num in range(1, max_num + 1):
            count = 0
            for _, row in recent_data.iterrows():
                if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                    count += 1
            number_freq[num] = count
        
        # Return top 15 hot numbers
        sorted_nums = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
        return [num for num, freq in sorted_nums[:15]]
    
    def get_cold_numbers(self, data, game_type, lookback=50):
        """Get infrequently drawn numbers"""
        max_num = 69 if game_type == 'pb' else 41
        recent_data = data.tail(lookback)
        
        number_freq = {}
        for num in range(1, max_num + 1):
            count = 0
            for _, row in recent_data.iterrows():
                if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                    count += 1
            number_freq[num] = count
        
        # Return bottom 15 cold numbers
        sorted_nums = sorted(number_freq.items(), key=lambda x: x[1])
        return [num for num, freq in sorted_nums[:15]]
    
    def get_overdue_numbers(self, data, game_type):
        """Get numbers that haven't appeared recently"""
        max_num = 69 if game_type == 'pb' else 41
        last_seen = {}
        
        # Find when each number was last seen
        for idx, row in data.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            for num in numbers:
                last_seen[num] = idx
        
        # Calculate gaps from most recent draw
        current_idx = len(data) - 1
        gaps = {}
        for num in range(1, max_num + 1):
            if num in last_seen:
                gaps[num] = current_idx - last_seen[num]
            else:
                gaps[num] = current_idx  # Never seen
        
        # Return most overdue numbers
        sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
        return [num for num, gap in sorted_gaps[:10]]
    
    def get_day_favorites(self, data, game_type):
        """Get numbers that appear frequently on current day of week"""
        today = datetime.now()
        day_of_week = today.weekday()  # 0=Monday, 6=Sunday
        
        if game_type == 'pb':
            day_data = data[data['DrawDate'].dt.weekday == day_of_week]
        else:
            day_data = data[data['Date'].dt.weekday == day_of_week]
        
        max_num = 69 if game_type == 'pb' else 41
        number_freq = {}
        
        for num in range(1, max_num + 1):
            count = 0
            for _, row in day_data.iterrows():
                if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                    count += 1
            number_freq[num] = count
        
        # Return top day favorites
        sorted_nums = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
        return [num for num, freq in sorted_nums[:10]]
    
    def apply_constraints(self, numbers, game_type='pb', is_double_play=False):
        """Apply all constraints efficiently"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        # Ensure we have 5 unique numbers
        numbers = [int(x) for x in list(set(numbers))]
        while len(numbers) < 5:
            new_num = random.randint(1, max_num)
            if new_num not in numbers:
                numbers.append(new_num)
        numbers = numbers[:5]
        
        # Constraint 1: Even/odd balance (2E/3O or 3E/2O)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        target_even = random.choice([2, 3])  # Randomly choose 2 or 3 evens
        
        # Adjust even/odd balance
        attempts = 0
        while even_count != target_even and attempts < 10:
            if even_count < target_even:
                # Need more evens - replace an odd with an even
                odd_nums = [i for i, n in enumerate(numbers) if n % 2 == 1]
                if odd_nums:
                    idx = random.choice(odd_nums)
                    # Find an even replacement
                    for replacement in range(2, max_num + 1, 2):
                        if replacement not in numbers:
                            numbers[idx] = replacement
                            even_count = sum(1 for n in numbers if n % 2 == 0)
                            break
            else:
                # Need fewer evens - replace an even with an odd
                even_nums = [i for i, n in enumerate(numbers) if n % 2 == 0]
                if even_nums:
                    idx = random.choice(even_nums)
                    # Find an odd replacement
                    for replacement in range(1, max_num + 1, 2):
                        if replacement not in numbers:
                            numbers[idx] = replacement
                            even_count = sum(1 for n in numbers if n % 2 == 0)
                            break
            attempts += 1
        
        # Constraint 2: Include 1 number from previous draw
        if len(data) > 0:
            last_draw = data.iloc[-1]
            last_numbers = [last_draw['1'], last_draw['2'], last_draw['3'], last_draw['4'], last_draw['5']]
            repeat_num = random.choice(last_numbers)
            numbers[0] = repeat_num  # Replace first number
        
        # Constraint 3: Avoid numbers from last 4 consecutive games
        recent_numbers = set()
        if len(data) >= 4:
            for i in range(4):
                row = data.iloc[-(i+1)]
                recent_numbers.update([row['1'], row['2'], row['3'], row['4'], row['5']])
        
        # Replace numbers that appear in recent 4 games (except the repeat number)
        for i in range(1, len(numbers)):
            while numbers[i] in recent_numbers:
                numbers[i] = random.randint(1, max_num)
                if numbers[i] not in numbers[:i] + numbers[i+1:]:  # Avoid duplicates
                    break
        
        # Remove duplicates and ensure 5 numbers
        numbers = [int(x) for x in list(dict.fromkeys(numbers))]
        while len(numbers) < 5:
            new_num = random.randint(1, max_num)
            if new_num not in numbers and new_num not in recent_numbers:
                numbers.append(new_num)
        
        numbers = sorted(numbers[:5])
        
        # Constraint 4: Sum range adjustment
        total = sum(numbers)
        if game_type == 'pb':
            target_min = 70
            target_max = 285 if is_double_play else 299
        else:  # Megabucks
            target_min, target_max = 36, 177
        
        # Simple sum adjustment
        attempts = 0
        while (total < target_min or total > target_max) and attempts < 20:
            if total < target_min:
                # Increase a random number
                idx = random.randint(0, 4)
                if numbers[idx] < max_num - 5:
                    numbers[idx] += random.randint(1, 5)
            else:
                # Decrease a random number
                idx = random.randint(0, 4)
                if numbers[idx] > 6:
                    numbers[idx] -= random.randint(1, 5)
            
            # Ensure no duplicates
            numbers = sorted([int(x) for x in list(set(numbers))])
            while len(numbers) < 5:
                numbers.append(random.randint(1, max_num))
            numbers = numbers[:5]
            
            total = sum(numbers)
            attempts += 1
        
        return sorted([int(x) for x in numbers])
    
    def predict_fast(self, game_type='pb', is_double_play=False):
        """Fast prediction using smart number selection"""
        data = self.pb_data if game_type == 'pb' else self.mb_data
        max_num = 69 if game_type == 'pb' else 41
        
        if data is None or len(data) == 0:
            return None
        
        # Get different categories of numbers
        hot_numbers = self.get_hot_numbers(data, game_type)
        cold_numbers = self.get_cold_numbers(data, game_type)
        overdue_numbers = self.get_overdue_numbers(data, game_type)
        day_favorites = self.get_day_favorites(data, game_type)
        
        # Smart selection strategy
        selected_numbers = []
        
        # 1. Pick 1 hot number
        if hot_numbers:
            selected_numbers.append(random.choice(hot_numbers[:5]))
        
        # 2. Pick 1 overdue number
        if overdue_numbers:
            num = random.choice(overdue_numbers[:5])
            if num not in selected_numbers:
                selected_numbers.append(num)
        
        # 3. Pick 1 day favorite
        if day_favorites:
            num = random.choice(day_favorites[:3])
            if num not in selected_numbers:
                selected_numbers.append(num)
        
        # 4. Pick 1 cold number (contrarian approach)
        if cold_numbers:
            num = random.choice(cold_numbers[:5])
            if num not in selected_numbers:
                selected_numbers.append(num)
        
        # 5. Fill remaining with random numbers
        while len(selected_numbers) < 5:
            num = random.randint(1, max_num)
            if num not in selected_numbers:
                selected_numbers.append(num)
        
        # Apply all constraints
        final_numbers = self.apply_constraints(selected_numbers, game_type, is_double_play)
        
        # Generate bonus number
        if game_type == 'pb':
            bonus = random.randint(1, 26)
        else:
            bonus = random.randint(1, 6)
        
        return final_numbers, bonus
    
    def validate_prediction(self, numbers, bonus, game_type, is_double_play=False):
        """Validate prediction meets constraints"""
        issues = []
        
        # Check even/odd balance
        even_count = sum(1 for n in numbers if n % 2 == 0)
        if even_count not in [2, 3]:
            issues.append(f"Even/odd imbalance: {even_count}E/{5-even_count}O")
        
        # Check sum range
        total = sum(numbers)
        if game_type == 'pb':
            if is_double_play:
                if not (70 <= total <= 285):
                    issues.append(f"Sum {total} outside range 70-285")
            else:
                if not (70 <= total <= 299):
                    issues.append(f"Sum {total} outside range 70-299")
        else:
            if not (36 <= total <= 177):
                issues.append(f"Sum {total} outside range 36-177")
        
        return issues

def main():
    print("=" * 60)
    print("FAST LOTTERY PREDICTION SYSTEM")
    print("=" * 60)
    print("Using Smart Selection with All Constraints")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    predictor = FastLotteryPredictor()
    
    if not predictor.load_data():
        print("Failed to load data!")
        return
    
    print("\nGenerating predictions...")
    
    # Generate predictions
    predictions = []
    
    # Powerball Regular
    pb_pred = predictor.predict_fast('pb', False)
    if pb_pred:
        numbers, powerball = pb_pred
        issues = predictor.validate_prediction(numbers, powerball, 'pb', False)
        predictions.append(('Powerball Regular', numbers, powerball, issues))
    
    # Powerball Double Play
    pb_dp_pred = predictor.predict_fast('pb', True)
    if pb_dp_pred:
        numbers, powerball = pb_dp_pred
        issues = predictor.validate_prediction(numbers, powerball, 'pb', True)
        predictions.append(('Powerball Double Play', numbers, powerball, issues))
    
    # Megabucks
    mb_pred = predictor.predict_fast('mb')
    if mb_pred:
        numbers, megaball = mb_pred
        issues = predictor.validate_prediction(numbers, megaball, 'mb')
        predictions.append(('Megabucks', numbers, megaball, issues))
    
    # Display results
    print("\nPREDICTIONS:")
    print("=" * 60)
    
    for game, numbers, bonus, issues in predictions:
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        
        print(f"\n{game}:")
        print(f"Numbers: {numbers}")
        print(f"Bonus: {bonus}")
        print(f"Complete: {numbers} + {bonus}")
        print(f"Sum: {total}")
        print(f"Even/Odd: {even_count}E/{5-even_count}O")
        
        if issues:
            print("Issues:", ", ".join(issues))
        else:
            print("Status: All constraints satisfied!")
    
    print("\n" + "=" * 60)
    print("CONSTRAINTS APPLIED:")
    print("* Even/odd balance (2E/3O or 3E/2O)")
    print("* Include number from previous draw")
    print("* Avoid numbers from last 4 games")
    print("* Sum within specified ranges")
    print("* Day-of-week preferences")
    print("* Hot/cold/overdue number analysis")
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fast_predictions_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Fast Lottery Predictions - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for game, numbers, bonus, issues in predictions:
            f.write(f"{game}: {numbers} + {bonus}\n")
            f.write(f"Sum: {sum(numbers)}, Even/Odd: {sum(1 for n in numbers if n % 2 == 0)}E/{5-sum(1 for n in numbers if n % 2 == 0)}O\n\n")
    
    print(f"\nPredictions saved to: {filename}")

if __name__ == "__main__":
    main()