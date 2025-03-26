import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CrimePredictionModel:
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
            
        # Load dataset
        self.df = pd.read_csv(dataset_path)
        
        # Validate required columns
        required_columns = ['District', 'Crime Rate (per 1,000 people)']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        
        # Prepare data
        self.X = self.df.drop(['District', 'Crime Rate (per 1,000 people)'], axis=1)
        self.y = self.df['Crime Rate (per 1,000 people)']
        
        # Validate data
        if self.X.empty or self.y.empty:
            raise ValueError("Empty dataset after preparation")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        self.model = LinearRegression()
        self.model.fit(self.X_train_scaled, self.y_train)
        
    def predict(self, input_data):
        if input_data is None or len(input_data) == 0:
            raise ValueError("Input data cannot be empty")
            
        # Validate input shape
        expected_features = len(self.X.columns)
        if input_data.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {input_data.shape[1]}")
            
        # Scale input data
        input_scaled = self.scaler.transform(input_data)
        return self.model.predict(input_scaled)
    
    def get_model_performance(self):
        # Predict and calculate R-squared
        y_pred = self.model.predict(self.X_test_scaled)
        r2 = r2_score(self.y_test, y_pred)
        return r2
    
    def create_visualizations(self, output_dir):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            feature_importance = pd.Series(
                abs(self.model.coef_), 
                index=self.X.columns
            ).sort_values(ascending=False)
            
            sns.barplot(x=feature_importance.values, y=feature_importance.index)
            plt.title('Feature Importance in Crime Rate Prediction')
            plt.xlabel('Absolute Coefficient Value')
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'feature_importance.png')
            plt.savefig(output_path)
            plt.close()
            
            return output_path
        except Exception as e:
            raise RuntimeError(f"Failed to create visualization: {str(e)}")

# Usage example in views
# model = CrimePredictionModel('dataset.csv')
# performance = model.get_model_performance()
# model.create_visualizations('predictor_app/static/images/')