from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
import pandas as pd
import numpy as np
from .ml_model import CrimePredictionModel
import os
from django.conf import settings

def get_model():
    # Initialize model with correct dataset path
    dataset_path = os.path.join(settings.BASE_DIR, 'dataset.csv')
    model = CrimePredictionModel(dataset_path)
    
    # Generate and save feature importance plot
    static_dir = os.path.join(settings.BASE_DIR, 'predictor_app', 'static', 'images')
    os.makedirs(static_dir, exist_ok=True)
    model.create_visualizations(static_dir)
    
    return model

def home(request):
    try:
        # Get model performance
        ml_model = get_model()
        r2_score = ml_model.get_model_performance()
        
        # Get unique districts from dataset
        dataset_path = os.path.join(settings.BASE_DIR, 'dataset.csv')
        districts = pd.read_csv(dataset_path)['District'].unique().tolist()
        
        context = {
            'r2_score': round(r2_score * 100, 2),
            'districts': districts
        }
        return render(request, 'predictor/home.html', context)
    except Exception as e:
        context = {
            'error': f"An error occurred: {str(e)}"
        }
        return render(request, 'predictor/home.html', context)

@csrf_protect
def predict(request):
    if request.method == 'POST':
        try:
            # Collect input data with validation
            input_fields = ['unemployment_rate', 'population_density', 'police_stations', 
                          'education_index', 'poverty_rate']
            
            input_data = []
            for field in input_fields:
                value = request.POST.get(field)
                if value is None or value.strip() == '':
                    raise ValueError(f"Missing required field: {field}")
                try:
                    input_data.append(float(value))
                except ValueError:
                    raise ValueError(f"Invalid value for {field}: {value}")
            
            # Convert to numpy array and reshape
            input_data = np.array([input_data])
            
            # Predict
            ml_model = get_model()
            prediction = ml_model.predict(input_data)[0]
            
            context = {
                'prediction': round(prediction, 2),
                'success': True
            }
        except Exception as e:
            context = {
                'error': f"Prediction failed: {str(e)}",
                'success': False
            }
        
        return render(request, 'predictor/predict.html', context)
    
    return render(request, 'predictor/predict.html')