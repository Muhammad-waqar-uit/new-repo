import pandas as pd
import numpy as np
from datetime import datetime
from enhanced_predictor import EnhancedLotteryPredictor
from lottery_predictor import LotteryPredictor
import warnings
warnings.filterwarnings('ignore')

class PredictionTester:
    def __init__(self):
        self.basic_predictor = LotteryPredictor()
        self.enhanced_predictor = EnhancedLotteryPredictor()
        
    def test_historical_performance(self, num_tests=20):
        """Test both predictors against historical data"""
        print("Loading data for testing...")
        
        # Load data for both predictors
        self.basic_predictor.load_data()
        self.enhanced_predictor.load_data()
        
        # Test Powerball
        print(f"\nTesting Powerball predictions on last {num_tests} draws...")
        self.test_game_performance('pb', num_tests)
        
        # Test Megabucks
        print(f"\nTesting Megabucks predictions on last {num_tests} draws...")
        self.test_game_performance('mb', num_tests)
    
    def test_game_performance(self, game_type, num_tests):
        """Test performance for a specific game"""
        if game_type == 'pb':
            data = self.basic_predictor.pb_data.copy()
        else:
            data = self.basic_predictor.mb_data.copy()
        
        if len(data) < num_tests + 20:
            print(f"Not enough data for testing {game_type}")
            return
        
        basic_results = []
        enhanced_results = []
        
        print(f"{'Draw':<4} {'Actual':<25} {'Basic Pred':<25} {'Enhanced Pred':<25} {'Basic Hits':<10} {'Enhanced Hits':<12}")
        print("-" * 110)
        
        for i in range(num_tests):
            test_idx = len(data) - num_tests + i
            
            # Get actual numbers
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
            
            # Prepare data up to test point
            if game_type == 'pb':
                self.basic_predictor.pb_data = data.iloc[:test_idx].copy()
                self.enhanced_predictor.pb_data = data.iloc[:test_idx].copy()
            else:
                self.basic_predictor.mb_data = data.iloc[:test_idx].copy()
                self.enhanced_predictor.mb_data = data.iloc[:test_idx].copy()
            
            # Train models
            try:
                self.basic_predictor.train_models()
                self.enhanced_predictor.analyze_patterns()
                self.enhanced_predictor.train_advanced_models()
                
                # Make predictions
                basic_pred = self.basic_predictor.predict(game_type)
                enhanced_pred = self.enhanced_predictor.predict_enhanced(game_type)
                
                if basic_pred and enhanced_pred:
                    basic_numbers, basic_bonus = basic_pred
                    enhanced_numbers, enhanced_bonus = enhanced_pred
                    
                    # Count matches
                    basic_matches = len(set(actual) & set(basic_numbers))
                    basic_bonus_match = 1 if actual_bonus == basic_bonus else 0
                    basic_total = basic_matches + basic_bonus_match
                    
                    enhanced_matches = len(set(actual) & set(enhanced_numbers))
                    enhanced_bonus_match = 1 if actual_bonus == enhanced_bonus else 0
                    enhanced_total = enhanced_matches + enhanced_bonus_match
                    
                    basic_results.append(basic_total)
                    enhanced_results.append(enhanced_total)
                    
                    # Format output
                    actual_str = f"{actual} + {actual_bonus}"
                    basic_str = f"{basic_numbers} + {basic_bonus}"
                    enhanced_str = f"{enhanced_numbers} + {enhanced_bonus}"
                    
                    print(f"{i+1:<4} {actual_str:<25} {basic_str:<25} {enhanced_str:<25} {basic_total:<10} {enhanced_total:<12}")
                    
            except Exception as e:
                print(f"Error in test {i+1}: {str(e)}")
                continue
        
        # Restore full data
        if game_type == 'pb':
            self.basic_predictor.pb_data = data
            self.enhanced_predictor.pb_data = data
        else:
            self.basic_predictor.mb_data = data
            self.enhanced_predictor.mb_data = data
        
        # Print summary
        if basic_results and enhanced_results:
            print("\n" + "=" * 110)
            print(f"PERFORMANCE SUMMARY FOR {game_type.upper()}")
            print("=" * 110)
            
            print(f"Basic Predictor:")
            print(f"  Average hits: {np.mean(basic_results):.2f}")
            print(f"  Best performance: {max(basic_results)} hits")
            print(f"  Tests with 2+ hits: {sum(1 for x in basic_results if x >= 2)}/{len(basic_results)} ({100*sum(1 for x in basic_results if x >= 2)/len(basic_results):.1f}%)")
            print(f"  Tests with 3+ hits: {sum(1 for x in basic_results if x >= 3)}/{len(basic_results)} ({100*sum(1 for x in basic_results if x >= 3)/len(basic_results):.1f}%)")
            
            print(f"\nEnhanced Predictor:")
            print(f"  Average hits: {np.mean(enhanced_results):.2f}")
            print(f"  Best performance: {max(enhanced_results)} hits")
            print(f"  Tests with 2+ hits: {sum(1 for x in enhanced_results if x >= 2)}/{len(enhanced_results)} ({100*sum(1 for x in enhanced_results if x >= 2)/len(enhanced_results):.1f}%)")
            print(f"  Tests with 3+ hits: {sum(1 for x in enhanced_results if x >= 3)}/{len(enhanced_results)} ({100*sum(1 for x in enhanced_results if x >= 3)/len(enhanced_results):.1f}%)")
            
            # Statistical comparison
            improvement = np.mean(enhanced_results) - np.mean(basic_results)
            print(f"\nImprovement: {improvement:+.2f} average hits")
            
            if improvement > 0:
                print("✓ Enhanced predictor shows improvement!")
            else:
                print("⚠ Enhanced predictor needs further tuning")
    
    def analyze_constraint_compliance(self):
        """Analyze how well predictions comply with constraints"""
        print("\nAnalyzing constraint compliance...")
        
        # Load data
        self.enhanced_predictor.load_data()
        self.enhanced_predictor.analyze_patterns()
        self.enhanced_predictor.train_advanced_models()
        
        # Test multiple predictions
        compliance_results = {
            'even_odd_balance': 0,
            'sum_range_pb': 0,
            'sum_range_pb_dp': 0,
            'sum_range_mb': 0,
            'no_all_even_odd': 0
        }
        
        num_tests = 100
        
        for i in range(num_tests):
            # Powerball regular
            pb_pred = self.enhanced_predictor.predict_enhanced('pb', False)
            if pb_pred:
                numbers, powerball = pb_pred
                even_count = sum(1 for n in numbers if n % 2 == 0)
                total = sum(numbers)
                
                # Check constraints
                if even_count in [2, 3]:
                    compliance_results['even_odd_balance'] += 1
                
                if 70 <= total <= 299:
                    compliance_results['sum_range_pb'] += 1
                
                if even_count not in [0, 5]:
                    compliance_results['no_all_even_odd'] += 1
            
            # Powerball double play
            pb_dp_pred = self.enhanced_predictor.predict_enhanced('pb', True)
            if pb_dp_pred:
                numbers, powerball = pb_dp_pred
                total = sum(numbers)
                
                if 70 <= total <= 285:
                    compliance_results['sum_range_pb_dp'] += 1
            
            # Megabucks
            mb_pred = self.enhanced_predictor.predict_enhanced('mb')
            if mb_pred:
                numbers, megaball = mb_pred
                total = sum(numbers)
                
                if 36 <= total <= 177:
                    compliance_results['sum_range_mb'] += 1
        
        print(f"\nConstraint Compliance Results (out of {num_tests} tests):")
        print(f"Even/Odd Balance (2E/3O or 3E/2O): {compliance_results['even_odd_balance']}/{num_tests} ({100*compliance_results['even_odd_balance']/num_tests:.1f}%)")
        print(f"No All Even/Odd: {compliance_results['no_all_even_odd']}/{num_tests} ({100*compliance_results['no_all_even_odd']/num_tests:.1f}%)")
        print(f"Powerball Sum Range (70-299): {compliance_results['sum_range_pb']}/{num_tests} ({100*compliance_results['sum_range_pb']/num_tests:.1f}%)")
        print(f"Powerball DP Sum Range (70-285): {compliance_results['sum_range_pb_dp']}/{num_tests} ({100*compliance_results['sum_range_pb_dp']/num_tests:.1f}%)")
        print(f"Megabucks Sum Range (36-177): {compliance_results['sum_range_mb']}/{num_tests} ({100*compliance_results['sum_range_mb']/num_tests:.1f}%)")

def main():
    print("Lottery Prediction Testing and Validation")
    print("=" * 50)
    
    tester = PredictionTester()
    
    # Test historical performance
    tester.test_historical_performance(15)
    
    # Analyze constraint compliance
    tester.analyze_constraint_compliance()
    
    print("\n" + "=" * 50)
    print("Testing completed!")
    print("The enhanced predictor uses advanced ML techniques to improve hit rates.")
    print("Results show performance against actual historical lottery draws.")

if __name__ == "__main__":
    main()