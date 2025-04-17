from fuel_efficiency_model import FuelEfficiencyPredictor

def test_model():
    # Initialize the predictor
    predictor = FuelEfficiencyPredictor()
    
    # Load the saved model (or train if not available)
    if not predictor.load_saved_model():
        print("Training new model...")
        predictor.train_model()
        predictor.save_model()
    
    # Test with some sample data
    test_cases = [
        {   # Chevrolet Chevelle Malibu
            'cylinders': 8,
            'displacement': 307.0,
            'horsepower': 130.0,
            'weight': 3504.0,
            'acceleration': 12.0,
            'model_year': 70,
            'origin': 1
        },
        {   # Toyota Corolla
            'cylinders': 4,
            'displacement': 97.0,
            'horsepower': 75.0,
            'weight': 2171.0,
            'acceleration': 16.0,
            'model_year': 75,
            'origin': 3
        },
        {   # Ford Mustang
            'cylinders': 6,
            'displacement': 250.0,
            'horsepower': 88.0,
            'weight': 3139.0,
            'acceleration': 14.5,
            'model_year': 71,
            'origin': 1
        }
    ]
    
    print("\nTesting the model with sample vehicles:")
    for i, car in enumerate(test_cases, 1):
        mpg = predictor.predict(car)
        print(f"Car {i}: Predicted MPG = {mpg:.1f}")

if __name__ == "__main__":
    test_model()