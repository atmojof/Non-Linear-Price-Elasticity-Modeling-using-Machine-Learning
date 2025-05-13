import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import shap
import matplotlib.pyplot as plt
import joblib

class PriceElasticityModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.elasticity = None
        
    def load_data(self, sales_path, products_path):
        sales = pd.read_csv(sales_path, parse_dates=['date'])
        products = pd.read_csv(products_path)
        df = sales.merge(products, on='item_id')
        
        # Feature engineering
        df['price_ratio'] = df.groupby(['item_id', 'store_id'])['unit_price'].transform(
            lambda x: x / x.rolling(7, min_periods=1).mean()
        )
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        return df.dropna()

    def create_pipeline(self):
        numeric_features = ['unit_price', 'price_ratio', 'pack_size']
        categorical_features = ['store_id', 'category']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100))
        ])

    def objective(self, trial, X, y):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'alpha': trial.suggest_float('alpha', 0, 10)
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = self.create_pipeline()
            model.set_params(regressor__**params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            scores.append(mean_squared_error(y_val, preds, squared=False))
            
        return np.mean(scores)

    def fit(self, sales_path, products_path):
        df = self.load_data(sales_path, products_path)
        X = df.drop(columns=['sales_quantity', 'date'])
        y = df['sales_quantity']
        
        # Hyperparameter optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=50)
        
        # Train final model
        self.model = self.create_pipeline()
        self.model.set_params(regressor__**study.best_params)
        self.model.fit(X, y)
        
        # Calculate price elasticity using SHAP
        preprocessed_data = self.model.named_steps['preprocessor'].transform(X)
        if hasattr(self.model.named_steps['regressor'], 'feature_names_in_'):
            self.feature_names = self.model.named_steps['regressor'].get_feature_names_out()
        else:
            self.feature_names = numeric_features + list(
                self.model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
            
        explainer = shap.Explainer(self.model.named_steps['regressor'])
        shap_values = explainer(preprocessed_data)
        
        price_idx = np.where([name == 'unit_price' for name in self.feature_names])[0][0]
        self.elasticity = np.mean(shap_values.values[:, price_idx] * X['unit_price'].mean() / y.mean())
        
        # Save artifacts
        joblib.dump(self.model, 'model/price_elasticity_model.pkl')
        df.to_csv('data/processed_data.csv', index=False)
        
    def predict(self, data):
        return self.model.predict(data)
    
    def plot_results(self):
        df = pd.read_csv('data/processed_data.csv')
        preds = self.model.predict(df.drop(columns=['sales_quantity', 'date']))
        
        plt.figure(figsize=(14, 6))
        plt.plot(df['date'], df['sales_quantity'], label='Actual Sales', alpha=0.7)
        plt.plot(df['date'], preds, label='Predicted Sales', alpha=0.7)
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales Quantity')
        plt.legend()
        plt.savefig('results/sales_predictions.png')
        plt.close()
        
        # SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, preprocessed_data, feature_names=self.feature_names)
        plt.savefig('results/feature_importance.png')
