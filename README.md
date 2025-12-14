# Batter Decision Value Modeling

A machine learning framework for evaluating MLB batters' swing decisions using Statcast data and expected run value models.

## Overview

This project quantifies the quality of batters' swing decisions by predicting the expected change in run expectancy for taking versus swinging at each pitch. The model incorporates pitch location, count context, and batter-specific "nitro zones" (areas where batters make optimal contact) to generate a comprehensive decision value metric.

## Key Features

- **Dual Model Approach**: Separate regression models for swing and take outcomes
- **Nitro Zone Analysis**: Convex hull-based identification of each batter's optimal contact zones
- **Called Strike Prediction**: Binary classification model to estimate strike probability
- **Decision Value Metric**: Year-over-year stable metric quantifying swing decision quality
- **Comprehensive Visualization**: Heatmaps, density plots, and player-specific analyses

## Methodology

### Data Processing

The analysis uses MLB Statcast data from 2021-2024, including:
- Pitch location (plate_x, plate_z)
- Count context (balls-strikes)
- Pitch outcomes and run expectancy changes
- Individual batter performance metrics

### Models

1. **Take Model** (LightGBM): Predicts delta run expectancy when not swinging
2. **Swing Model** (LightGBM): Predicts delta run expectancy when swinging
3. **Called Strike Model** (XGBoost): Estimates probability of called strike

### Nitro Zone Calculation

For each batter, the nitro zone represents the convex hull of pitch locations where they achieved their top 5% exit velocities on balls in play.

### Decision Value Formula

```
Decision Value = (Swing Expectancy × Strike Probability) - (Take Expectancy × Ball Probability)
```

Player-level metrics are standardized to a 100-scale with standard deviation of 10.

## Results

### Model Performance

**Swing Model (LightGBM)**
- Training RMSE: 0.2977
- Testing RMSE: 0.2963

**Take Model (LightGBM)**
- Training RMSE: 0.0438
- Testing RMSE: 0.0446

**Called Strike Model (XGBoost)**
- AUC: 0.9816
- Accuracy: 93%

## Technologies

- **Python** - Core programming language
- **pybaseball** - MLB Statcast data retrieval
- **XGBoost** - Gradient boosting for called strike prediction
- **LightGBM** - Gradient boosting for run expectancy models
- **scikit-learn** - Model evaluation and preprocessing
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization
- **scipy** - Statistical analysis and convex hull computation
- **pingouin** - Statistical testing
- **adjustText** - Plot label optimization

## Project Structure

```
batter_decision_value/
├── data_fetch.ipynb              # Statcast data retrieval (2021-2024)
├── batter_decision_value_v3.ipynb # Main analysis and modeling
├── batter_decision_value_v2.ipynb # Previous iteration
├── batter_decision_value.ipynb   # Initial version
└── README.md
```

## Usage

1. **Data Collection**: Run [data_fetch.ipynb](data_fetch.ipynb) to retrieve latest Statcast data
2. **Analysis**: Execute [batter_decision_value_v3.ipynb](batter_decision_value_v3.ipynb) for complete modeling pipeline
3. **Results**: View player rankings and decision value metrics in the final cells

## Insights

- Decision value effectively captures swing discipline quality
- Nitro zones vary significantly by batter, reflecting individual swing mechanics
- Count context heavily influences optimal decision-making
- Top decision-makers show 20+ point advantage over league average

## Future Improvements

- Incorporate pitcher tendencies and pitch type
- Add temporal features (inning, score differential)
- Develop real-time decision recommendation system
- Extend to defensive positioning optimization

## Data Source

All data sourced from MLB Advanced Media via the pybaseball package.