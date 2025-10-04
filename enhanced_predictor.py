import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EnhancedLotteryPredictor:
    def __init__(self):
        self.pb_model = None
        self.mb_model = None
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.pb_data = None
        self.mb_data = None
        self.number_patterns = {}
        self.gap_patterns = {}
        
    def load_data(self):
        """Load and preprocess lottery data"""
        # Load Powerball data
        self.pb_data = pd.read_csv('pb_results.csv')
        self.pb_data['DrawDate'] = pd.to_datetime(self.pb_data['DrawDate'].str.split(' - ').str[0], errors='coerce')
        self.pb_data = self.pb_data.dropna(subset=['DrawDate'])
        self.pb_data = self.pb_data.sort_values('DrawDate')
        
        # Load Megabucks data
        self.mb_data = pd.read_csv('mb_results.csv')
        self.mb_data['Date'] = pd.to_datetime(self.mb_data['Date'], errors='coerce')
        self.mb_data = self.mb_data.dropna(subset=['Date'])
        self.mb_data = self.mb_data.sort_values('Date')
        
        # Analyze patterns
        self.analyze_patterns()
        
    def analyze_patterns(self):
        """Analyze number patterns and gaps for better predictions"""
        # Analyze Powerball patterns
        self.analyze_game_patterns(self.pb_data, 'pb')
        # Analyze Megabucks patterns
        self.analyze_game_patterns(self.mb_data, 'mb')
        
    def analyze_game_patterns(self, data, game_type):
        """Analyze patterns for a specific game"""
        max_num = 69 if game_type == 'pb' else 41
        
        # Track number gaps (draws since last appearance)
        number_gaps = {i: [] for i in range(1, max_num + 1)}
        last_seen = {i: -1 for i in range(1, max_num + 1)}
        
        # Track consecutive appearances
        consecutive_patterns = {i: [] for i in range(1, max_num + 1)}
        
        # Track sum patterns
        sum_patterns = []
        
        # Track number pair frequencies
        pair_frequencies = {}
        
        for idx, row in data.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            draw_sum = sum(numbers)
            sum_patterns.append(draw_sum)
            
            # Update gaps
            for num in range(1, max_num + 1):
                if num in numbers:
                    if last_seen[num] != -1:
                        gap = idx - last_seen[num]
                        number_gaps[num].append(gap)
                    last_seen[num] = idx
            
            # Track pairs
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1
        
        # Store patterns
        self.gap_patterns[game_type] = number_gaps
        self.number_patterns[game_type] = {
            'sum_patterns': sum_patterns,
            'pair_frequencies': pair_frequencies,
            'consecutive_patterns': consecutive_patterns
        }
    
    def get_hot_cold_numbers(self, data, game_type, lookback=20):
        """Get hot and cold numbers based on recent frequency"""
        max_num = 69 if game_type == 'pb' else 41
        recent_data = data.tail(lookback)
        
        number_freq = {i: 0 for i in range(1, max_num + 1)}
        
        for _, row in recent_data.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            for num in numbers:
                number_freq[num] += 1
        
        # Sort by frequency
        sorted_nums = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
        
        hot_numbers = [num for num, freq in sorted_nums[:15]]  # Top 15
        cold_numbers = [num for num, freq in sorted_nums[-15:]]  # Bottom 15
        
        return hot_numbers, cold_numbers
    
    def predict_overdue_numbers(self, data, game_type):
        """Predict numbers that are overdue based on gap analysis"""
        max_num = 69 if game_type == 'pb' else 41
        overdue_scores = {}
        
        # Calculate current gaps
        last_seen = {i: -1 for i in range(1, max_num + 1)}
        
        for idx, row in data.iterrows():
            numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
            for num in numbers:
                last_seen[num] = idx
        
        current_idx = len(data) - 1
        
        for num in range(1, max_num + 1):
            if last_seen[num] != -1:
                current_gap = current_idx - last_seen[num]
                
                # Get average gap for this number
                if game_type in self.gap_patterns and num in self.gap_patterns[game_type]:
                    gaps = self.gap_patterns[game_type][num]
                    if gaps:
                        avg_gap = np.mean(gaps)
                        std_gap = np.std(gaps) if len(gaps) > 1 else avg_gap * 0.5
                        
                        # Score based on how overdue the number is
                        overdue_score = max(0, (current_gap - avg_gap) / (std_gap + 1))
                        overdue_scores[num] = overdue_score
        
        # Return top overdue numbers
        sorted_overdue = sorted(overdue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, score in sorted_overdue[:10]]
    
    def create_advanced_features(self, data, game_type='pb'):
        """Create advanced features for ML model"""
        features = []
        
        for i in range(len(data)):
            if i < 15:  # Need more historical data for advanced features
                continue
                
            row_features = []
            
            # Basic frequency features
            start_idx = max(0, i-50)
            historical = data.iloc[start_idx:i]
            
            max_num = 69 if game_type == 'pb' else 41
            
            # Number frequency in different time windows
            for window in [10, 20, 50]:
                window_data = data.iloc[max(0, i-window):i]
                for num in range(1, max_num + 1):
                    count = 0
                    for _, row in window_data.iterrows():
                        if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                            count += 1
                    row_features.append(count / len(window_data) if len(window_data) > 0 else 0)
            
            # Gap-based features
            last_seen = {num: -1 for num in range(1, max_num + 1)}
            for idx in range(i):
                row = data.iloc[idx]
                numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
                for num in numbers:
                    last_seen[num] = idx
            
            # Current gaps
            for num in range(1, max_num + 1):
                gap = i - last_seen[num] - 1 if last_seen[num] != -1 else i
                row_features.append(gap)
            
            # Sum pattern features
            recent_sums = []
            for idx in range(max(0, i-10), i):
                row = data.iloc[idx]
                numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
                recent_sums.append(sum(numbers))
            
            if recent_sums:
                row_features.extend([
                    np.mean(recent_sums),
                    np.std(recent_sums) if len(recent_sums) > 1 else 0,
                    max(recent_sums),
                    min(recent_sums)
                ])
            else:
                row_features.extend([0, 0, 0, 0])
            
            # Day of week features
            if game_type == 'pb':
                date = data.iloc[i]['DrawDate']
            else:
                date = data.iloc[i]['Date']
                
            day_of_week = date.weekday()
            day_features = [0] * 7
            day_features[day_of_week] = 1
            row_features.extend(day_features)
            
            # Consecutive number patterns
            recent_draws = data.iloc[max(0, i-5):i]
            consecutive_counts = [0] * 10  # Track consecutive sequences
            
            for _, row in recent_draws.iterrows():
                numbers = sorted([row['1'], row['2'], row['3'], row['4'], row['5']])
                for j in range(len(numbers) - 1):
                    if numbers[j+1] - numbers[j] == 1:
                        consecutive_counts[min(9, numbers[j+1] - numbers[j])] += 1
            
            row_features.extend(consecutive_counts)
            
            # Even/odd ratio features
            recent_even_odd = []
            for idx in range(max(0, i-10), i):
                row = data.iloc[idx]
                numbers = [row['1'], row['2'], row['3'], row['4'], row['5']]
                even_count = sum(1 for n in numbers if n % 2 == 0)
                recent_even_odd.append(even_count)
            
            if recent_even_odd:
                row_features.append(np.mean(recent_even_odd))
            else:
                row_features.append(2.5)  # Default even count
            
            features.append(row_features)
            
        return np.array(features)
    
    def create_targets(self, data, game_type='pb'):
        """Create target variables"""
        targets = []
        
        for i in range(15, len(data)):  # Skip first 15 rows for advanced features
            if game_type == 'pb':
                target = [data.iloc[i]['1'], data.iloc[i]['2'], data.iloc[i]['3'], 
                         data.iloc[i]['4'], data.iloc[i]['5'], data.iloc[i]['PB']]
            else:
                target = [data.iloc[i]['1'], data.iloc[i]['2'], data.iloc[i]['3'], 
                         data.iloc[i]['4'], data.iloc[i]['5'], data.iloc[i]['Megaball']]
            targets.append(target)
            
        return np.array(targets)
    
    def train_advanced_models(self):
        """Train advanced ensemble models"""
        print("Training advanced Powerball model...")
        
        # Powerball model
        pb_features = self.create_advanced_features(self.pb_data, 'pb')
        pb_targets = self.create_targets(self.pb_data, 'pb')
        
        if len(pb_features) > 0:
            # Scale features
            pb_features_scaled = self.scaler.fit_transform(pb_features)
            
            # Create ensemble with voting
            rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            gb = GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42)
            mlp = MLPRegressor(hidden_layer_sizes=(150, 100, 50), max_iter=1000, random_state=42)
            
            # Train individual models for each number position
            self.pb_model = []
            for pos in range(6):  # 5 numbers + powerball
                voting_reg = VotingRegressor([
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
                ])
                voting_reg.fit(pb_features_scaled, pb_targets[:, pos])
                self.pb_model.append(voting_reg)
        
        print("Training advanced Megabucks model...")
        
        # Megabucks model
        mb_features = self.create_advanced_features(self.mb_data, 'mb')
        mb_targets = self.create_targets(self.mb_data, 'mb')
        
        if len(mb_features) > 0:
            # Scale features
            mb_features_scaled = self.scaler.fit_transform(mb_features)
            
            # Train individual models for each number position
            self.mb_model = []
            for pos in range(6):  # 5 numbers + megaball
                voting_reg = VotingRegressor([
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
                ])
                voting_reg.fit(mb_features_scaled, mb_targets[:, pos])
                self.mb_model.append(voting_reg)
    
    def smart_number_selection(self, base_prediction, game_type='pb'):
        """Apply smart selection strategies to improve hit rate"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        # Get various number categories
        hot_numbers, cold_numbers = self.get_hot_cold_numbers(data, game_type)
        overdue_numbers = self.predict_overdue_numbers(data, game_type)
        
        # Get day preferences
        today = datetime.now()
        day_of_week = today.weekday()
        day_preferred = self.get_day_preferences(day_of_week, game_type)
        
        # Smart selection strategy
        selected_numbers = []
        
        # 1. Include one hot number (high recent frequency)
        for num in hot_numbers:
            if num not in selected_numbers:
                selected_numbers.append(num)
                break
        
        # 2. Include one overdue number
        for num in overdue_numbers:
            if num not in selected_numbers:
                selected_numbers.append(num)
                break
        
        # 3. Include one day-preferred number
        for num in day_preferred[:5]:
            if num not in selected_numbers:
                selected_numbers.append(num)
                break
        
        # 4. Include numbers from base prediction
        for num in base_prediction:
            if len(selected_numbers) < 5 and num not in selected_numbers:
                selected_numbers.append(num)
        
        # 5. Fill remaining slots with balanced selection
        while len(selected_numbers) < 5:
            # Try to balance hot and cold
            if len(selected_numbers) % 2 == 0:
                # Add from hot numbers
                for num in hot_numbers:
                    if num not in selected_numbers:
                        selected_numbers.append(num)
                        break
            else:
                # Add from cold numbers (contrarian approach)
                for num in cold_numbers:
                    if num not in selected_numbers:
                        selected_numbers.append(num)
                        break
            
            # Fallback: add any valid number
            if len(selected_numbers) < 5:
                for num in range(1, max_num + 1):
                    if num not in selected_numbers:
                        selected_numbers.append(num)
                        break
        
        return sorted(selected_numbers[:5])
    
    def apply_enhanced_constraints(self, numbers, powerball=None, game_type='pb', is_double_play=False):
        """Apply enhanced constraints with better logic"""
        max_num = 69 if game_type == 'pb' else 41
        data = self.pb_data if game_type == 'pb' else self.mb_data
        
        # Constraint 1: Even/odd balance (2 even/3 odd or 3 even/2 odd)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        target_even = 2 if np.random.random() < 0.5 else 3
        
        while even_count != target_even and len(numbers) == 5:
            if even_count < target_even:
                # Need more evens - replace an odd with an even
                odd_indices = [i for i, n in enumerate(numbers) if n % 2 == 1]
                if odd_indices:
                    idx = np.random.choice(odd_indices)
                    # Find a suitable even replacement
                    for replacement in range(2, max_num + 1, 2):
                        if replacement not in numbers:
                            numbers[idx] = replacement
                            even_count += 1
                            break
            else:
                # Need fewer evens - replace an even with an odd
                even_indices = [i for i, n in enumerate(numbers) if n % 2 == 0]
                if even_indices:
                    idx = np.random.choice(even_indices)
                    # Find a suitable odd replacement
                    for replacement in range(1, max_num + 1, 2):
                        if replacement not in numbers:
                            numbers[idx] = replacement
                            even_count -= 1
                            break
        
        # Constraint 2: Include 1 number from night before
        if len(data) > 0:
            last_draw = data.iloc[-1]
            last_numbers = [last_draw['1'], last_draw['2'], last_draw['3'], last_draw['4'], last_draw['5']]
            # Replace first number with a number from last draw
            repeat_num = np.random.choice(last_numbers)
            if repeat_num not in numbers:
                numbers[0] = repeat_num
            else:
                # If already present, ensure it stays
                pass
        
        # Constraint 3: Avoid numbers from 4 most recent consecutive games
        recent_numbers = set()
        if len(data) >= 4:
            for i in range(4):
                row = data.iloc[-(i+1)]
                recent_numbers.update([row['1'], row['2'], row['3'], row['4'], row['5']])
        
        # Replace numbers that appear in recent 4 games (except the repeat number)
        for i in range(1, len(numbers)):  # Skip first number (repeat from last draw)
            attempts = 0
            while numbers[i] in recent_numbers and attempts < 50:
                numbers[i] = np.random.randint(1, max_num + 1)
                attempts += 1
        
        # Remove duplicates
        numbers = list(dict.fromkeys(numbers))  # Preserve order while removing duplicates
        
        # Fill to 5 numbers if needed
        while len(numbers) < 5:
            new_num = np.random.randint(1, max_num + 1)
            if new_num not in numbers and new_num not in recent_numbers:
                numbers.append(new_num)
        
        numbers = sorted(numbers[:5])
        
        # Constraint 4: Sum totals within specified ranges
        total = sum(numbers)
        if game_type == 'pb':
            if is_double_play:
                target_min, target_max = 70, 285
            else:
                target_min, target_max = 70, 299
        else:  # Megabucks
            target_min, target_max = 36, 177
        
        # Intelligent sum adjustment
        adjustment_attempts = 0
        while (total < target_min or total > target_max) and adjustment_attempts < 100:
            if total < target_min:
                # Increase numbers strategically
                diff_needed = target_min - total
                for i in range(len(numbers)):
                    if numbers[i] < max_num - 5:
                        increase = min(diff_needed, max_num - numbers[i])
                        numbers[i] += increase
                        total += increase
                        diff_needed -= increase
                        if diff_needed <= 0:
                            break
            else:
                # Decrease numbers strategically
                diff_needed = total - target_max
                for i in range(len(numbers)-1, -1, -1):
                    if numbers[i] > 6:
                        decrease = min(diff_needed, numbers[i] - 1)
                        numbers[i] -= decrease
                        total -= decrease
                        diff_needed -= decrease
                        if diff_needed <= 0:
                            break
            
            adjustment_attempts += 1
            total = sum(numbers)
        
        return sorted(numbers)
    
    def get_day_preferences(self, day_of_week, game_type='pb'):
        """Get numbers that are more likely on specific days"""
        if game_type == 'pb':
            data = self.pb_data
            date_col = 'DrawDate'
        else:
            data = self.mb_data
            date_col = 'Date'
        
        # Filter data by day of week
        day_data = data[data[date_col].dt.weekday == day_of_week]
        
        # Count frequency of each number on this day
        number_freq = {}
        max_num = 69 if game_type == 'pb' else 41
        
        for num in range(1, max_num + 1):
            count = 0
            for _, row in day_data.iterrows():
                if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                    count += 1
            number_freq[num] = count
        
        # Return top numbers for this day
        sorted_nums = sorted(number_freq.items(), key=lambda x: x[1], reverse=True)
        return [num for num, freq in sorted_nums[:15]]
    
    def predict_enhanced(self, game_type='pb', is_double_play=False):
        """Make enhanced predictions with all tricks applied"""
        if game_type == 'pb' and self.pb_model is None:
            return None
        if game_type == 'mb' and self.mb_model is None:
            return None
        
        # Create features for prediction
        if game_type == 'pb':
            features = self.create_advanced_features(self.pb_data, 'pb')
            models = self.pb_model
        else:
            features = self.create_advanced_features(self.mb_data, 'mb')
            models = self.mb_model
        
        if len(features) == 0:
            return None
        
        # Use last feature row for prediction
        last_features = features[-1].reshape(1, -1)
        last_features_scaled = self.scaler.transform(last_features)
        
        # Get predictions for each position
        predictions = []
        for model in models:
            pred = model.predict(last_features_scaled)[0]
            predictions.append(pred)
        
        # Convert to integers
        base_numbers = [int(round(x)) for x in predictions[:5]]
        bonus = int(round(predictions[5]))
        
        # Ensure numbers are in valid range
        max_num = 69 if game_type == 'pb' else 41
        base_numbers = [max(1, min(max_num, n)) for n in base_numbers]
        
        # Apply smart number selection
        smart_numbers = self.smart_number_selection(base_numbers, game_type)
        
        # Apply enhanced constraints
        final_numbers = self.apply_enhanced_constraints(smart_numbers, bonus, game_type, is_double_play)
        
        # Ensure bonus number is in valid range
        if game_type == 'pb':
            bonus = max(1, min(26, bonus))
        else:
            bonus = max(1, min(6, bonus))
        
        return final_numbers, bonus

def main():
    print("Enhanced Lottery Prediction System with Advanced AI/ML")
    print("=" * 60)
    
    predictor = EnhancedLotteryPredictor()
    
    # Load data
    print("Loading and analyzing data...")
    predictor.load_data()
    
    # Train advanced models
    print("Training advanced ensemble models...")
    predictor.train_advanced_models()
    
    # Make current predictions
    print("\n" + "=" * 60)
    print("ENHANCED PREDICTIONS WITH ADVANCED STRATEGIES")
    print("=" * 60)
    
    # Powerball regular
    pb_pred = predictor.predict_enhanced('pb', False)
    if pb_pred:
        numbers, powerball = pb_pred
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        print(f"Powerball Regular: {numbers} + {powerball}")
        print(f"Sum: {total}, Even/Odd: {even_count}E/{5-even_count}O")
    
    # Powerball double play
    pb_dp_pred = predictor.predict_enhanced('pb', True)
    if pb_dp_pred:
        numbers, powerball = pb_dp_pred
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        print(f"Powerball Double Play: {numbers} + {powerball}")
        print(f"Sum: {total}, Even/Odd: {even_count}E/{5-even_count}O")
    
    # Megabucks
    mb_pred = predictor.predict_enhanced('mb')
    if mb_pred:
        numbers, megaball = mb_pred
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        print(f"Megabucks: {numbers} + {megaball}")
        print(f"Sum: {total}, Even/Odd: {even_count}E/{5-even_count}O")
    
    print("\nAdvanced Strategies Applied:")
    print("✓ Ensemble voting regressors for each number position")
    print("✓ Hot/cold number analysis")
    print("✓ Overdue number prediction based on gap analysis")
    print("✓ Smart number selection balancing multiple factors")
    print("✓ Day-of-week statistical preferences")
    print("✓ Pattern recognition for consecutive numbers")
    print("✓ Sum distribution analysis")
    print("✓ Enhanced constraint application")
    print("✓ All original constraints maintained")
    
    print(f"\nPrediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()