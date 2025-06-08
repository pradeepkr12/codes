import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Any, Callable
from datetime import datetime
import logging
from abc import ABC, abstractmethod

# Statistical/Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HoltWinters
try:
    from pmdarima import auto_arima
except ImportError:
    print("pmdarima not found. Install with: pip install pmdarima")
    auto_arima = None

# Machine Learning Libraries
try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_percentage_error
except ImportError:
    print("XGBoost or sklearn not found. Install with: pip install xgboost scikit-learn")
    xgb = None

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

class BaseModel(ABC):
    """Abstract base class for forecasting models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> None:
        """Fit the model to training data"""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts for specified number of steps"""
        pass
    
    def validate_data(self, data: pd.Series) -> bool:
        """Basic data validation"""
        if data.isnull().any():
            logging.warning(f"Data contains null values for {self.name}")
            return False
        if len(data) < 12:  # Minimum data requirement
            logging.warning(f"Insufficient data for {self.name} (need at least 12 points)")
            return False
        return True

class AutoARIMAModel(BaseModel):
    """Auto ARIMA implementation"""
    
    def __init__(self):
        super().__init__("Auto ARIMA")
        
    def fit(self, data: pd.Series, **kwargs) -> None:
        if not self.validate_data(data) or auto_arima is None:
            return
        
        try:
            self.model = auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=True,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                **kwargs
            )
            self.fitted = True
        except Exception as e:
            logging.error(f"Auto ARIMA fitting failed: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted or self.model is None:
            return np.full(steps, np.nan)
        
        try:
            forecast = self.model.predict(n_periods=steps)
            return np.maximum(forecast, 0)  # Ensure non-negative forecasts
        except Exception as e:
            logging.error(f"Auto ARIMA prediction failed: {e}")
            return np.full(steps, np.nan)

class BATSModel(BaseModel):
    """BATS (Box-Cox transform, ARMA errors, Trend and Seasonal components)"""
    
    def __init__(self):
        super().__init__("BATS")
        
    def fit(self, data: pd.Series, **kwargs) -> None:
        if not self.validate_data(data):
            return
        
        try:
            # Using Exponential Smoothing as approximation for BATS
            self.model = ExponentialSmoothing(
                data,
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                **kwargs
            ).fit()
            self.fitted = True
        except Exception as e:
            logging.error(f"BATS fitting failed: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted or self.model is None:
            return np.full(steps, np.nan)
        
        try:
            forecast = self.model.forecast(steps)
            return np.maximum(forecast, 0)
        except Exception as e:
            logging.error(f"BATS prediction failed: {e}")
            return np.full(steps, np.nan)

class TBATSModel(BaseModel):
    """TBATS (Trigonometric seasonality, Box-Cox transform, ARMA errors, Trend and Seasonal components)"""
    
    def __init__(self):
        super().__init__("TBATS")
        
    def fit(self, data: pd.Series, **kwargs) -> None:
        if not self.validate_data(data):
            return
        
        try:
            # Using Holt-Winters as approximation for TBATS
            self.model = HoltWinters(
                data,
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                **kwargs
            ).fit()
            self.fitted = True
        except Exception as e:
            logging.error(f"TBATS fitting failed: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted or self.model is None:
            return np.full(steps, np.nan)
        
        try:
            forecast = self.model.forecast(steps)
            return np.maximum(forecast, 0)
        except Exception as e:
            logging.error(f"TBATS prediction failed: {e}")
            return np.full(steps, np.nan)

class CrostonModel(BaseModel):
    """Croston's method for intermittent demand"""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__("Croston")
        self.alpha = alpha
        self.demand_avg = None
        self.interval_avg = None
        
    def fit(self, data: pd.Series, **kwargs) -> None:
        if not self.validate_data(data):
            return
        
        try:
            non_zero_demand = data[data > 0]
            intervals = []
            
            last_demand_idx = -1
            for i, val in enumerate(data):
                if val > 0:
                    if last_demand_idx >= 0:
                        intervals.append(i - last_demand_idx)
                    last_demand_idx = i
            
            if len(non_zero_demand) > 0 and len(intervals) > 0:
                self.demand_avg = non_zero_demand.mean()
                self.interval_avg = np.mean(intervals)
                self.fitted = True
            else:
                self.fitted = False
        except Exception as e:
            logging.error(f"Croston fitting failed: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted:
            return np.full(steps, np.nan)
        
        try:
            forecast_per_period = self.demand_avg / self.interval_avg
            return np.full(steps, max(forecast_per_period, 0))
        except Exception as e:
            logging.error(f"Croston prediction failed: {e}")
            return np.full(steps, np.nan)

class XGBoostModel(BaseModel):
    """XGBoost time series forecasting model"""
    
    def __init__(self, lags: int = 12):
        super().__init__("XGBoost")
        self.lags = lags
        self.scaler = StandardScaler() if 'StandardScaler' in globals() else None
        self.last_values = None
        
    def create_features(self, data: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for XGBoost"""
        X, y = [], []
        
        for i in range(self.lags, len(data)):
            X.append(data.iloc[i-self.lags:i].values)
            y.append(data.iloc[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series, **kwargs) -> None:
        if not self.validate_data(data) or xgb is None:
            return
        
        try:
            X, y = self.create_features(data)
            if len(X) == 0:
                self.fitted = False
                return
            
            # Scale features if scaler is available
            if self.scaler is not None:
                X = self.scaler.fit_transform(X)
            
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                **kwargs
            )
            self.model.fit(X, y)
            self.last_values = data.iloc[-self.lags:].values
            self.fitted = True
        except Exception as e:
            logging.error(f"XGBoost fitting failed: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> np.ndarray:
        if not self.fitted or self.model is None:
            return np.full(steps, np.nan)
        
        try:
            forecasts = []
            current_window = self.last_values.copy()
            
            for _ in range(steps):
                X_pred = current_window.reshape(1, -1)
                if self.scaler is not None:
                    X_pred = self.scaler.transform(X_pred)
                
                next_pred = self.model.predict(X_pred)[0]
                forecasts.append(max(next_pred, 0))
                
                # Update window for next prediction
                current_window = np.append(current_window[1:], next_pred)
            
            return np.array(forecasts)
        except Exception as e:
            logging.error(f"XGBoost prediction failed: {e}")
            return np.full(steps, np.nan)

class ForecastingFramework:
    """Main forecasting framework"""
    
    def __init__(self):
        self.models = {
            'auto_arima': AutoARIMAModel(),
            'bats': BATSModel(),
            'tbats': TBATSModel(),
            'croston': CrostonModel(),
            'xgboost': XGBoostModel()
        }
        self.metrics = {
            'mape': self._calculate_mape,
            'mae': self._calculate_mae,
            'rmse': self._calculate_rmse
        }
        self.results = {}
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a new model to the framework"""
        self.models[name] = model
    
    def add_metric(self, name: str, metric_func: Callable) -> None:
        """Add a new evaluation metric"""
        self.metrics[name] = metric_func
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Avoid division by zero
        mask = actual != 0
        if not mask.any():
            return np.inf
        
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def _calculate_mae(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(actual - predicted))
    
    def _calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def prepare_data(self, data: pd.DataFrame, validation_months: int = 3) -> Dict[str, Dict]:
        """
        Prepare training and validation datasets for each SKU
        
        Args:
            data: DataFrame with SKUs as columns and dates as index
            validation_months: Number of months for validation
        
        Returns:
            Dictionary with SKU data splits
        """
        prepared_data = {}
        
        for sku in data.columns:
            sku_data = data[sku].dropna()
            
            if len(sku_data) < validation_months + 12:  # Minimum data requirement
                logging.warning(f"Insufficient data for SKU {sku}")
                continue
            
            # Split data
            train_data = sku_data.iloc[:-validation_months]
            validation_data = sku_data.iloc[-validation_months:]
            
            prepared_data[sku] = {
                'train': train_data,
                'validation': validation_data,
                'full_data': sku_data
            }
        
        return prepared_data
    
    def evaluate_models(self, sku_data: Dict, primary_metric: str = 'mape') -> Dict:
        """
        Evaluate all models on validation data for a single SKU
        
        Args:
            sku_data: Dictionary containing train, validation, and full data
            primary_metric: Primary metric for model selection
        
        Returns:
            Dictionary with model performance results
        """
        train_data = sku_data['train']
        validation_data = sku_data['validation']
        validation_steps = len(validation_data)
        
        model_results = {}
        
        for model_name, model in self.models.items():
            try:
                # Fit model
                model.fit(train_data)
                
                if not model.fitted:
                    model_results[model_name] = {
                        'fitted': False,
                        'error': 'Model fitting failed'
                    }
                    continue
                
                # Generate predictions
                predictions = model.predict(validation_steps)
                
                if np.isnan(predictions).all():
                    model_results[model_name] = {
                        'fitted': False,
                        'error': 'Prediction failed'
                    }
                    continue
                
                # Calculate metrics
                metrics_results = {}
                for metric_name, metric_func in self.metrics.items():
                    try:
                        metrics_results[metric_name] = metric_func(
                            validation_data.values, predictions
                        )
                    except Exception as e:
                        metrics_results[metric_name] = np.inf
                        logging.error(f"Metric {metric_name} calculation failed: {e}")
                
                model_results[model_name] = {
                    'fitted': True,
                    'predictions': predictions,
                    'metrics': metrics_results
                }
                
            except Exception as e:
                logging.error(f"Model {model_name} evaluation failed: {e}")
                model_results[model_name] = {
                    'fitted': False,
                    'error': str(e)
                }
        
        return model_results
    
    def select_best_model(self, model_results: Dict, primary_metric: str = 'mape') -> str:
        """Select the best performing model based on primary metric"""
        best_model = None
        best_score = np.inf
        
        for model_name, results in model_results.items():
            if not results.get('fitted', False):
                continue
            
            score = results['metrics'].get(primary_metric, np.inf)
            if score < best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def generate_final_forecast(self, sku_data: Dict, best_model_name: str, 
                              forecast_months: int = 5) -> np.ndarray:
        """Generate final forecast using the best model on full data"""
        if best_model_name is None or best_model_name not in self.models:
            return np.full(forecast_months, np.nan)
        
        full_data = sku_data['full_data']
        best_model = self.models[best_model_name]
        
        try:
            # Refit on full data
            best_model.fit(full_data)
            
            if best_model.fitted:
                return best_model.predict(forecast_months)
            else:
                return np.full(forecast_months, np.nan)
        except Exception as e:
            logging.error(f"Final forecast generation failed: {e}")
            return np.full(forecast_months, np.nan)
    
    def run_forecasting_pipeline(self, data: pd.DataFrame, 
                               validation_months: int = 3,
                               forecast_months: int = 5,
                               primary_metric: str = 'mape') -> Dict:
        """
        Run the complete forecasting pipeline
        
        Args:
            data: Input data with SKUs as columns
            validation_months: Months for validation
            forecast_months: Months to forecast
            primary_metric: Primary evaluation metric
        
        Returns:
            Complete results dictionary
        """
        logging.info("Starting forecasting pipeline...")
        
        # Prepare data
        prepared_data = self.prepare_data(data, validation_months)
        logging.info(f"Prepared data for {len(prepared_data)} SKUs")
        
        pipeline_results = {}
        
        for sku, sku_data in prepared_data.items():
            logging.info(f"Processing SKU: {sku}")
            
            # Evaluate models
            model_results = self.evaluate_models(sku_data, primary_metric)
            
            # Select best model
            best_model = self.select_best_model(model_results, primary_metric)
            
            # Generate final forecast
            if best_model:
                final_forecast = self.generate_final_forecast(
                    sku_data, best_model, forecast_months
                )
            else:
                final_forecast = np.full(forecast_months, np.nan)
                logging.warning(f"No suitable model found for SKU {sku}")
            
            pipeline_results[sku] = {
                'model_results': model_results,
                'best_model': best_model,
                'final_forecast': final_forecast,
                'validation_data': sku_data['validation'].values
            }
        
        self.results = pipeline_results
        logging.info("Forecasting pipeline completed!")
        return pipeline_results
    
    def get_summary_report(self) -> pd.DataFrame:
        """Generate a summary report of all results"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for sku, results in self.results.items():
            best_model = results['best_model']
            
            if best_model and best_model in results['model_results']:
                best_metrics = results['model_results'][best_model]['metrics']
                
                summary_data.append({
                    'SKU': sku,
                    'Best_Model': best_model,
                    'MAPE': best_metrics.get('mape', np.nan),
                    'MAE': best_metrics.get('mae', np.nan),
                    'RMSE': best_metrics.get('rmse', np.nan),
                    'Forecast_Mean': np.nanmean(results['final_forecast']),
                    'Forecast_Std': np.nanstd(results['final_forecast'])
                })
            else:
                summary_data.append({
                    'SKU': sku,
                    'Best_Model': 'None',
                    'MAPE': np.nan,
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'Forecast_Mean': np.nan,
                    'Forecast_Std': np.nan
                })
        
        return pd.DataFrame(summary_data)

# Example usage
if __name__ == "__main__":
    # Create sample data (replace with your actual data)
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='MS')
    np.random.seed(42)
    
    # Generate sample SKU data with trends and seasonality
    n_skus = 5
    sample_data = {}
    
    for i in range(n_skus):
        trend = np.linspace(100, 200, len(dates))
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 10, len(dates))
        intermittent = np.random.binomial(1, 0.8, len(dates))  # 80% non-zero
        
        values = (trend + seasonal + noise) * intermittent
        values = np.maximum(values, 0)  # Ensure non-negative
        
        sample_data[f'SKU_{i+1}'] = values
    
    df = pd.DataFrame(sample_data, index=dates)
    
    # Initialize and run forecasting framework
    framework = ForecastingFramework()
    
    # Run the pipeline
    results = framework.run_forecasting_pipeline(
        data=df,
        validation_months=3,
        forecast_months=5,
        primary_metric='mape'
    )
    
    # Get summary report
    summary = framework.get_summary_report()
    print("\nSummary Report:")
    print(summary.to_string(index=False))
    
    # Example of how to add a new model
    class SimpleMovingAverage(BaseModel):
        def __init__(self, window=3):
            super().__init__("Simple MA")
            self.window = window
            self.last_values = None
        
        def fit(self, data: pd.Series, **kwargs):
            if len(data) >= self.window:
                self.last_values = data.iloc[-self.window:].values
                self.fitted = True
            else:
                self.fitted = False
        
        def predict(self, steps: int) -> np.ndarray:
            if not self.fitted:
                return np.full(steps, np.nan)
            
            forecast_value = np.mean(self.last_values)
            return np.full(steps, max(forecast_value, 0))
    
    # Add the new model
    framework.add_model('simple_ma', SimpleMovingAverage())
    
    # Example of adding a new metric
    def mean_absolute_scaled_error(actual, predicted):
        """Calculate Mean Absolute Scaled Error"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Calculate naive forecast error (using previous period)
        naive_error = np.mean(np.abs(np.diff(actual)))
        if naive_error == 0:
            return np.inf
        
        mae = np.mean(np.abs(actual - predicted))
        return mae / naive_error
    
    framework.add_metric('mase', mean_absolute_scaled_error)
    
    print("\nFramework is ready with additional model and metric!")
    print(f"Available models: {list(framework.models.keys())}")
    print(f"Available metrics: {list(framework.metrics.keys())}")
