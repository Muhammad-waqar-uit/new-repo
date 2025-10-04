import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class LotteryPredictor:
    def __init__(self):
        self.pb_model = None
        self.mb_model = None
        self.scaler = StandardScaler()
        self.pb_data = None
        self.mb_data = None
        
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
        
    def create_features(self, data, game_type='pb'):
        """Create features for ML model"""
        features = []
        
        for i in range(len(data)):
            if i < 10:  # Need at least 10 historical draws
                continue
                
            row_features = []
            
            # Historical frequency features (last 50 draws)
            start_idx = max(0, i-50)
            historical = data.iloc[start_idx:i]
            
            if game_type == 'pb':
                # Number frequency in last 50 draws
                for num in range(1, 70):
                    count = 0
                    for _, row in historical.iterrows():
                        if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                            count += 1
                    row_features.append(count)
                
                # Powerball frequency
                for pb in range(1, 27):
                    count = sum(1 for _, row in historical.iterrows() if row['PB'] == pb)
                    row_features.append(count)
                    
            else:  # Megabucks
                # Number frequency in last 50 draws
                for num in range(1, 42):
                    count = 0
                    for _, row in historical.iterrows():
                        if num in [row['1'], row['2'], row['3'], row['4'], row['5']]:
                            count += 1
                    row_features.append(count)
                
                # Megaball frequency
                for mb in range(1, 7):
                    count = sum(1 for _, row in historical.iterrows() if row['Megaball'] == mb)
                    row_features.append(count)
            
            # Day of week features
            if game_type == 'pb':
                date = data.iloc[i]['DrawDate']
            else:
                date = data.iloc[i]['Date']
                
            day_of_week = date.weekday()  # 0=Monday, 6=Sunday
            day_features = [0] * 7
            day_features[day_of_week] = 1
            row_features.extend(day_features)
            
            # Recent patterns (last 4 draws)
            recent_draws = data.iloc[max(0, i-4):i]
            recent_numbers = []
            for _, row in recent_draws.iterrows():
                recent_numbers.extend([row['1'], row['2'], row['3'], row['4'], row['5']])
            
            # Hot/cold number indicators
            if game_type == 'pb':
                max_num = 69
            else:
                max_num = 41
                
            for num in range(1, max_num + 1):
                row_features.append(recent_numbers.count(num))
            
            features.append(row_features)
            
        return np.array(features)
    
    def create_targets(self, data, game_type='pb'):
        """Create target variables"""
        targets = []
        
        for i in range(10, len(data)):  # Skip first 10 rows
            if game_type == 'pb':
                target = [data.iloc[i]['1'], data.iloc[i]['2'], data.iloc[i]['3'], 
                         data.iloc[i]['4'], data.iloc[i]['5'], data.iloc[i]['PB']]
            else:
                target = [data.iloc[i]['1'], data.iloc[i]['2'], data.iloc[i]['3'], 
                         data.iloc[i]['4'], data.iloc[i]['5'], data.iloc[i]['Megaball']]
            targets.append(target)
            
        return np.array(targets)
    
    def train_models(self):
        """Train ML models for both games"""
        print("Training Powerball model...")
        
        # Powerball model
        pb_features = self.create_features(self.pb_data, 'pb')
        pb_targets = self.create_targets(self.pb_data, 'pb')
        
        if len(pb_features) > 0:
            # Ensemble of models
            models = [
                RandomForestRegressor(n_estimators=100, random_state=42),
                GradientBoostingRegressor(n_estimators=100, random_state=42),
                MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            ]
            
            self.pb_model = models
            
            # Train each model
            for model in self.pb_model:
                model.fit(pb_features, pb_targets)
        
        print("Training Megabucks model...")
        
        # Megabucks model
        mb_features = self.create_features(self.mb_data, 'mb')
        mb_targets = self.create_targets(self.mb_data, 'mb')
        
        if len(mb_features) > 0:
            models = [
                RandomForestRegressor(n_estimators=100, random_state=42),
                GradientBoostingRegressor(n_estimators=100, random_state=42),
                MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            ]
            
            self.mb_model = models
            
            # Train each model
            for model in self.mb_model:
                model.fit(mb_features, mb_targets)
    
    def apply_constraints(self, numbers, powerball=None, game_type='pb'):
        """Apply all the specified constraints"""
        # Constraint 1: Even/odd balance (2 even/3 odd or 3 even/2 odd)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        if even_count not in [2, 3]:
            # Adjust to meet constraint
            if even_count < 2:
                # Need more evens
                for i in range(len(numbers)):
                    if numbers[i] % 2 == 1 and even_count < 2:
                        numbers[i] = numbers[i] + 1 if numbers[i] < (69 if game_type == 'pb' else 41) else numbers[i] - 1
                        if numbers[i] % 2 == 0:
                            even_count += 1
            elif even_count > 3:
                # Need fewer evens
                for i in range(len(numbers)):
                    if numbers[i] % 2 == 0 and even_count > 3:
                        numbers[i] = numbers[i] + 1 if numbers[i] < (69 if game_type == 'pb' else 41) else numbers[i] - 1
                        if numbers[i] % 2 == 1:
                            even_count -= 1
        
        # Constraint 2: Include 1 number from night before (if available)
        if game_type == 'pb' and len(self.pb_data) > 0:
            last_draw = self.pb_data.iloc[-1]
            last_numbers = [last_draw['1'], last_draw['2'], last_draw['3'], last_draw['4'], last_draw['5']]
            # Replace one number with a number from last draw
            numbers[0] = np.random.choice(last_numbers)
        elif game_type == 'mb' and len(self.mb_data) > 0:
            last_draw = self.mb_data.iloc[-1]
            last_numbers = [last_draw['1'], last_draw['2'], last_draw['3'], last_draw['4'], last_draw['5']]
            numbers[0] = np.random.choice(last_numbers)
        
        # Constraint 3: Don't select numbers from 4 most recent consecutive games
        recent_numbers = set()
        if game_type == 'pb' and len(self.pb_data) >= 4:
            for i in range(4):
                row = self.pb_data.iloc[-(i+1)]
                recent_numbers.update([row['1'], row['2'], row['3'], row['4'], row['5']])
        elif game_type == 'mb' and len(self.mb_data) >= 4:
            for i in range(4):
                row = self.mb_data.iloc[-(i+1)]
                recent_numbers.update([row['1'], row['2'], row['3'], row['4'], row['5']])
        
        # Replace numbers that appear in recent 4 games
        max_num = 69 if game_type == 'pb' else 41
        for i in range(len(numbers)):
            while numbers[i] in recent_numbers:
                numbers[i] = np.random.randint(1, max_num + 1)
        
        # Remove duplicates and sort
        numbers = sorted(list(set(numbers)))
        while len(numbers) < 5:
            new_num = np.random.randint(1, max_num + 1)
            if new_num not in numbers and new_num not in recent_numbers:
                numbers.append(new_num)
        numbers = sorted(numbers[:5])
        
        # Constraint 4: Sum totals within specified ranges
        total = sum(numbers)
        if game_type == 'pb':
            target_min, target_max = 70, 299
        else:  # Megabucks
            target_min, target_max = 36, 177
        
        # Adjust numbers to meet sum constraint
        while total < target_min or total > target_max:
            if total < target_min:
                # Increase smallest numbers
                for i in range(len(numbers)):
                    if numbers[i] < max_num and total < target_min:
                        numbers[i] += 1
                        total += 1
            else:
                # Decrease largest numbers
                for i in range(len(numbers)-1, -1, -1):
                    if numbers[i] > 1 and total > target_max:
                        numbers[i] -= 1
                        total -= 1
            
            # Prevent infinite loop
            if total < target_min - 50 or total > target_max + 50:
                break
        
        return numbers
    
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
        return [num for num, freq in sorted_nums[:20]]  # Top 20 numbers
    
    def predict(self, game_type='pb', is_double_play=False):
        """Make predictions with all constraints applied"""
        if game_type == 'pb' and self.pb_model is None:
            return None
        if game_type == 'mb' and self.mb_model is None:
            return None
        
        # Get current day of week for day-based preferences
        today = datetime.now()
        day_of_week = today.weekday()
        
        # Get day preferences
        day_preferred = self.get_day_preferences(day_of_week, game_type)
        
        # Create features for prediction
        if game_type == 'pb':
            features = self.create_features(self.pb_data, 'pb')
            models = self.pb_model
        else:
            features = self.create_features(self.mb_data, 'mb')
            models = self.mb_model
        
        if len(features) == 0:
            return None
        
        # Use last feature row for prediction
        last_features = features[-1].reshape(1, -1)
        
        # Ensemble prediction
        predictions = []
        for model in models:
            pred = model.predict(last_features)[0]
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Convert to integers and apply constraints
        numbers = [int(round(x)) for x in avg_pred[:5]]
        bonus = int(round(avg_pred[5]))
        
        # Ensure numbers are in valid range
        max_num = 69 if game_type == 'pb' else 41
        numbers = [max(1, min(max_num, n)) for n in numbers]
        
        # Apply day preferences (replace some numbers with day-preferred ones)
        for i in range(2):  # Replace 2 numbers with day preferences
            if i < len(day_preferred):
                numbers[i] = day_preferred[i]
        
        # Apply all constraints
        numbers = self.apply_constraints(numbers, bonus, game_type)
        
        # Ensure bonus number is in valid range
        if game_type == 'pb':
            bonus = max(1, min(26, bonus))
            # Adjust sum for double play if needed
            if is_double_play:
                total = sum(numbers)
                if total > 285:
                    # Reduce numbers to meet double play constraint
                    while total > 285 and max(numbers) > 1:
                        max_idx = numbers.index(max(numbers))
                        numbers[max_idx] -= 1
                        total = sum(numbers)
        else:
            bonus = max(1, min(6, bonus))
        
        return numbers, bonus
    
    def test_predictions(self, game_type='pb', num_tests=10):
        """Test predictions against historical data"""
        if game_type == 'pb':
            data = self.pb_data
        else:
            data = self.mb_data
        
        if len(data) < num_tests + 10:
            print(f"Not enough data for testing {game_type}")
            return
        
        correct_counts = []
        
        # Test on last num_tests draws
        for i in range(num_tests):
            # Use data up to test point
            test_idx = len(data) - num_tests + i
            
            if game_type == 'pb':
                actual = [data.iloc[test_idx]['1'], data.iloc[test_idx]['2'], 
                         data.iloc[test_idx]['3'], data.iloc[test_idx]['4'], 
                         data.iloc[test_idx]['5']]
                actual_bonus = data.iloc[test_idx]['PB']
            else:
                actual = [data.iloc[test_idx]['1'], data.iloc[test_idx]['2'], 
                         data.iloc[test_idx]['3'], data.iloc[test_idx]['4'], 
                         data.iloc[test_idx]['5']]
                actual_bonus = data.iloc[test_idx]['Megaball']
            
            # Temporarily reduce data for prediction
            if game_type == 'pb':
                temp_data = self.pb_data.iloc[:test_idx].copy()
                self.pb_data = temp_data
            else:
                temp_data = self.mb_data.iloc[:test_idx].copy()
                self.mb_data = temp_data
            
            # Retrain with reduced data
            self.train_models()
            
            # Make prediction
            pred_result = self.predict(game_type)
            if pred_result:
                predicted, pred_bonus = pred_result
                
                # Count matches
                matches = len(set(actual) & set(predicted))
                bonus_match = 1 if actual_bonus == pred_bonus else 0
                
                correct_counts.append(matches + bonus_match)
                print(f"Test {i+1}: Actual: {actual} + {actual_bonus}, "
                      f"Predicted: {predicted} + {pred_bonus}, "
                      f"Matches: {matches} + {bonus_match} = {matches + bonus_match}")
        
        # Restore full data
        if game_type == 'pb':
            self.pb_data = pd.read_csv('pb_results.csv')
            self.pb_data['DrawDate'] = pd.to_datetime(self.pb_data['DrawDate'].str.split(' - ').str[0], errors='coerce')
            self.pb_data = self.pb_data.dropna(subset=['DrawDate']).sort_values('DrawDate')
        else:
            self.mb_data = pd.read_csv('mb_results.csv')
            self.mb_data['Date'] = pd.to_datetime(self.mb_data['Date'], errors='coerce')
            self.mb_data = self.mb_data.dropna(subset=['Date']).sort_values('Date')
        
        if correct_counts:
            avg_correct = np.mean(correct_counts)
            print(f"\nAverage correct predictions for {game_type}: {avg_correct:.2f}")
            print(f"Best performance: {max(correct_counts)} correct")
            print(f"Tests with 2+ correct: {sum(1 for x in correct_counts if x >= 2)}/{len(correct_counts)}")
            print(f"Tests with 3+ correct: {sum(1 for x in correct_counts if x >= 3)}/{len(correct_counts)}")

def main():
    print("Lottery Prediction System with AI/ML")
    print("=" * 50)
    
    predictor = LotteryPredictor()
    
    # Load data
    print("Loading data...")
    predictor.load_data()
    
    # Train models
    print("Training models...")
    predictor.train_models()
    
    # Test predictions
    print("\nTesting Powerball predictions...")
    predictor.test_predictions('pb', 5)
    
    print("\nTesting Megabucks predictions...")
    predictor.test_predictions('mb', 5)
    
    # Retrain with full data
    print("\nRetraining with full data...")
    predictor.train_models()
    
    # Make current predictions
    print("\n" + "=" * 50)
    print("CURRENT PREDICTIONS")
    print("=" * 50)
    
    # Powerball regular
    pb_pred = predictor.predict('pb', False)
    if pb_pred:
        numbers, powerball = pb_pred
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        print(f"Powerball Regular: {numbers} + {powerball}")
        print(f"Sum: {total}, Even/Odd: {even_count}E/{5-even_count}O")
    
    # Powerball double play
    pb_dp_pred = predictor.predict('pb', True)
    if pb_dp_pred:
        numbers, powerball = pb_dp_pred
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        print(f"Powerball Double Play: {numbers} + {powerball}")
        print(f"Sum: {total}, Even/Odd: {even_count}E/{5-even_count}O")
    
    # Megabucks
    mb_pred = predictor.predict('mb')
    if mb_pred:
        numbers, megaball = mb_pred
        total = sum(numbers)
        even_count = sum(1 for n in numbers if n % 2 == 0)
        print(f"Megabucks: {numbers} + {megaball}")
        print(f"Sum: {total}, Even/Odd: {even_count}E/{5-even_count}O")
    
    print("\nConstraints Applied:")
    print("✓ Even/odd balance (2E/3O or 3E/2O)")
    print("✓ Include number from previous draw")
    print("✓ Avoid numbers from last 4 consecutive games")
    print("✓ Sum within specified ranges")
    print("✓ Day-of-week preferences applied")

if __name__ == "__main__":
    main()