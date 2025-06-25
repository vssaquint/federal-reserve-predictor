
def predict_fed_decision(economic_data):
    """
    Predict Federal Reserve policy decision based on economic indicators.
    
    Parameters:
    economic_data (dict or pd.Series): Economic indicators with feature names as keys
    
    Returns:
    dict: Prediction results with decision and confidence
    """
    import joblib
    import pandas as pd
    import numpy as np
    
    # Load model and preprocessing objects
    model = joblib.load('models/trained/final_model_random_forest.pkl')
    scaler = joblib.load('models/features/feature_scaler_cleaned.pkl')
    label_encoder = joblib.load('models/features/label_encoder_cleaned.pkl')
    
    # Expected features
    feature_names = ['gdp_growth_yoy_lag_6', 'UNRATE_lag_3', 'unemployment_change', 'DGS10', 'yield_curve_slope', 'inflation_yoy', 'gdp_growth_yoy_lag_3', 'fed_funds_momentum', 'real_fed_funds', 'CPIAUCSL', 'gdp_growth_yoy', 'employment_growth_yoy', 'gdp_growth_yoy_lag_1', 'employment_growth_yoy_lag_3']
    
    # Prepare input data
    if isinstance(economic_data, dict):
        input_df = pd.DataFrame([economic_data])
    else:
        input_df = pd.DataFrame([economic_data.to_dict()])
    
    # Ensure all features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Select and order features
    X = input_df[feature_names]
    
    # Apply scaling if required
    scaling_required = False
    if scaling_required:
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
    else:
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None
    
    # Decode prediction
    decision = label_encoder.inverse_transform([prediction])[0]
    
    # Prepare result
    result = {
        'decision': decision,
        'confidence': float(np.max(probabilities)) if probabilities is not None else None,
        'probabilities': {
            label_encoder.inverse_transform([i])[0]: float(prob) 
            for i, prob in enumerate(probabilities)
        } if probabilities is not None else None
    }
    
    return result
