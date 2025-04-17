import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os

class FuelEfficiencyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
        self.target = 'mpg'
        
    def load_data(self, file_path='auto-mpg.data'):
        """Load and preprocess the data"""
        try:
            # Read the data file with proper column names
            column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                          'acceleration', 'model_year', 'origin', 'car_name']
            data = pd.read_csv(file_path, delim_whitespace=True, header=None, 
                             names=column_names, na_values='?')
            
            # Drop car name as it's not useful for prediction
            data = data.drop('car_name', axis=1)
            
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data):
        """Handle missing values and scale features"""
        # Separate features and target
        X = data[self.features]
        y = data[self.target]
        
        # Handle missing values (horsepower has some)
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled, y
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the Random Forest model"""
        # Load and preprocess data
        data = self.load_data()
        if data is None:
            return False
        
        X, y = self.preprocess_data(data)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        
        # Initialize and train the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model trained successfully")
        print(f"Test MSE: {mse:.2f}")
        print(f"Test R²: {r2:.2f}")
        
        return True
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # Convert input to DataFrame if it's not already
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=self.features)
        
        # Preprocess the input data
        X_imputed = self.imputer.transform(input_data)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)
        return prediction[0]
    
    def save_model(self, model_path='fuel_efficiency_model.joblib'):
        """Save the trained model to disk"""
        if self.model is None:
            print("No model to save. Please train the model first.")
            return False
        
        # Save both the model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'features': self.features
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        return True
    
    def load_saved_model(self, model_path='fuel_efficiency_model.joblib'):
        """Load a previously saved model"""
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return False
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.features = model_data['features']
        
        print("Model loaded successfully")
        return True


if __name__ == "__main__":
    # Example usage
    predictor = FuelEfficiencyPredictor()
    
    # Train and save the model
    predictor.train_model()
    predictor.save_model()
    
    # Example prediction
    sample_car = {
        'cylinders': 8,
        'displacement': 307.0,
        'horsepower': 130.0,
        'weight': 3504.0,
        'acceleration': 12.0,
        'model_year': 70,
        'origin': 1
    }
    
    prediction = predictor.predict(sample_car)
    print(f"Predicted MPG: {prediction:.2f}")