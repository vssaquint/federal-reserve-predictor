# Federal Reserve Interest Rate Decision Predictor

## Overview

This project implements a machine learning model that predicts Federal Reserve interest rate decisions with 88.7% accuracy using 69 years of historical economic data. The model analyzes various economic indicators to predict whether the Federal Reserve will raise, lower, or maintain interest rates.

This project was developed as part of my Master's in Economics at Universidad de Granada.

## Project Structure

```
federal-reserve-predictor/
├── data/
│   ├── raw/                 # Original data files
│   └── processed/           # Cleaned and processed datasets
├── models/
│   ├── trained/            # Saved model files
│   └── features/           # Feature engineering artifacts
├── outputs/
│   ├── figures/            # Visualizations and plots
│   └── model_results/      # Model performance metrics
├── 01_data_collection.ipynb
├── 02_exploratory_analysis.ipynb
├── 03_feature_engineering.ipynb
└── 04_modeling.ipynb
```

## Methodology

### 1. Data Collection
The project utilizes 69 years of economic data including:
- Historical Federal Reserve interest rate decisions
- Core economic indicators
- Financial market data
- Macroeconomic variables

### 2. Exploratory Data Analysis
- Time series analysis of economic indicators
- Correlation analysis between features and rate decisions
- Identification of patterns during different economic cycles
- Crisis period analysis

### 3. Feature Engineering
- Creation of lagged variables to capture temporal patterns
- Economic indicator transformations
- Feature scaling and normalization
- Selection of most predictive features

### 4. Model Development
- Implementation of Random Forest classifier
- Hyperparameter optimization
- Cross-validation for robust performance evaluation
- Final model achieving 88.7% accuracy

## Key Features

- **High Accuracy**: 88.7% prediction accuracy on test data
- **Comprehensive Feature Set**: Utilizes multiple economic indicators
- **Robust Validation**: Time series cross-validation to prevent data leakage
- **Interpretable Results**: Feature importance analysis for economic insights

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vssaquint/federal-reserve-predictor.git
cd federal-reserve-predictor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Collection**: Run `01_data_collection.ipynb` to gather and prepare the raw data

2. **Exploratory Analysis**: Execute `02_exploratory_analysis.ipynb` to understand data patterns

3. **Feature Engineering**: Run `03_feature_engineering.ipynb` to create model features

4. **Model Training**: Execute `04_modeling.ipynb` to train and evaluate the model

## Model Performance

- **Accuracy**: 88.7%
- **Model Type**: Random Forest Classifier
- **Validation Method**: Time series cross-validation
- **Training Period**: 69 years of historical data

## Key Insights

The model identifies several important predictors of Federal Reserve decisions:
- Current inflation rates and trends
- Employment statistics
- GDP growth indicators
- Financial market conditions
- Previous rate decision patterns

## Future Improvements

- Incorporate real-time data feeds for live predictions
- Explore deep learning approaches for potentially higher accuracy
- Add economic regime detection for context-aware predictions
- Develop API for model deployment
- Create interactive dashboard for visualizations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Universidad de Granada, Master's in Economics program
- Federal Reserve Economic Data (FRED) for providing historical data
- Economic research papers that informed feature selection
- Open source community for the tools and libraries used

## Contact

For questions or feedback, please open an issue in this repository.
