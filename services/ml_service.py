# services/ml_service.py
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
import asyncio
import joblib
import numpy as np
import pandas as pd
import structlog
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler  # For clustering scaling
from config.settings import settings
from utils.exceptions import ModelTrainingError
logger = structlog.get_logger()
class MLService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_directory)
        self._setup_models()
    def _setup_models(self):
        """Initialize available models"""
        self.classification_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=settings.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=settings.random_state),
            'decision_tree': DecisionTreeClassifier(random_state=settings.random_state),
            'knn': KNeighborsClassifier(),
            'svm': SVC(probability=True, random_state=settings.random_state),
        }
        self.regression_models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=settings.random_state),
            'decision_tree': DecisionTreeRegressor(random_state=settings.random_state),
            'knn': KNeighborsRegressor(),
            'svm': SVR(),
        }
        self.clustering_models = {
            'kmeans': KMeans(n_clusters=3, random_state=settings.random_state, n_init=10),
            'dbscan': DBSCAN(),
        }
    def train_model(
        self,
        file_path: str,
        task_type: str,
        model_type: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train a machine learning model and return results (including metrics)."""
        try:
            start_time = time.time()
            # Load preprocessed data (assume CSV)
            df = pd.read_csv(file_path, encoding='utf-8', engine='python', on_bad_lines='skip')
            # Select default model if not specified
            if not model_type:
                model_type = self._get_default_model(task_type)
            # Train based on task type
            if task_type == "classification":
                result = self._train_classification(df, model_type, target_column)
            elif task_type == "regression":
                result = self._train_regression(df, model_type, target_column)
            elif task_type == "clustering":
                result = self._train_clustering(df, model_type)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            # Add timing information
            result["training_time"] = time.time() - start_time
            # Save model (defensively)
            if "model" in result:
                model_filename = f"trained_model_{Path(file_path).stem}.pkl"
                model_path = self.upload_dir / model_filename
                # ensure path exists
                self.upload_dir.mkdir(parents=True, exist_ok=True)
                try:
                    joblib.dump(result["model"], model_path)
                except Exception as e:
                    logger.error(f"Failed to save model to {model_path}: {str(e)}")
                    result.setdefault("error", "")
                    result["error"] = (result.get("error", "") + f" | Failed to save model: {str(e)}").lstrip(" | ")
                # Remove heavy object before returning
                if "model" in result:
                    del result["model"]
                # Save model metadata (features, plan sidecar link if available)
                try:
                    # features: infer from the preprocessed CSV header (exclude target column if present)
                    df_pre = pd.read_csv(file_path, nrows=1)
                    features = list(df_pre.columns)
                    # if target_column present, remove from feature list
                    if target_column and target_column in features:
                        features = [c for c in features if c != target_column]
                    metadata = {
                        "model_file": model_path.name,
                        "features": features,
                        "trained_on": Path(file_path).name,
                        "task_type": task_type,
                        "model_type": model_type,
                        "created_at": time.time(),
                        "metrics": result.get("results", {})
                    }
                    meta_path = model_path.with_suffix(".json")
                    with open(meta_path, "w", encoding="utf-8") as fh:
                        json.dump(metadata, fh, indent=2)
                except Exception as e:
                    logger.warning("Failed to save model metadata sidecar", error=str(e))
            logger.info(f"Model training completed: {task_type}/{model_type}", file=str(file_path))
            return result
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            # Wrap into consistent error response
            return {
                "task_type": task_type,
                "model_type": model_type,
                "results": {},
                "error": f"Training failed: {str(e)}"
            }
    def _train_classification(self, df: pd.DataFrame, model_type: str, target_column: str) -> Dict[str, Any]:
        """Train classification model"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        if model_type not in self.classification_models:
            raise ValueError(f"Unsupported classification model: {model_type}")
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.default_test_size, random_state=settings.random_state
        )
        # Train model
        model = self.classification_models[model_type]
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate metrics
        results = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        }
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        # Feature importance
        feature_importance = self._get_feature_importance(model, X.columns)
        return {
            "task_type": "classification",
            "model_type": model_type,
            "results": results,
            "feature_importance": feature_importance,
            "model": model
        }
    def _train_regression(self, df: pd.DataFrame, model_type: str, target_column: str) -> Dict[str, Any]:
        """Train regression model"""
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        if model_type not in self.regression_models:
            raise ValueError(f"Unsupported regression model: {model_type}")
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.default_test_size, random_state=settings.random_state
        )
        # Train model
        model = self.regression_models[model_type]
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate metrics
        results = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mean_squared_error": float(mean_squared_error(y_test, y_pred)),
            "mean_absolute_error": float(mean_absolute_error(y_test, y_pred))
        }
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"] = float(cv_scores.std())
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        # Feature importance
        feature_importance = self._get_feature_importance(model, X.columns)
        return {
            "task_type": "regression",
            "model_type": model_type,
            "results": results,
            "feature_importance": feature_importance,
            "model": model
        }
    def _train_clustering(self, df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """Train clustering model"""
        if model_type not in self.clustering_models:
            raise ValueError(f"Unsupported clustering model: {model_type}")
        # Prepare data (use all numeric columns for clustering)
        X = df.select_dtypes(include=[np.number]) # Only numeric columns
        if X.empty:
            raise ValueError("No numeric columns found for clustering")
        # Auto-scale for clustering (standard practice)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # Train model
        model = self.clustering_models[model_type]
        if hasattr(model, 'fit_predict'):
            labels = model.fit_predict(X_scaled)
        else:
            model.fit(X_scaled)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)
        # Calculate metrics
        results = {}
        unique_labels = np.unique(labels)
        # Exclude noise label (-1) for DBSCAN if present
        clean_labels = [l for l in unique_labels if l != -1]
        if len(clean_labels) > 1:
            try:
                results["silhouette_score"] = float(silhouette_score(X_scaled, labels))
                results["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
            except Exception as e:
                logger.warning(f"Clustering metrics calculation failed: {str(e)}")
        # Model-specific metrics
        if model_type == "kmeans":
            try:
                results["inertia"] = float(model.inertia_)
                results["n_clusters"] = int(model.n_clusters)
            except Exception:
                pass
        return {
            "task_type": "clustering",
            "model_type": model_type,
            "results": results,
            "model": model,
            "scaler": scaler  # For prediction alignment
        }
    def _get_default_model(self, task_type: str) -> str:
        """Get default model for task type"""
        defaults = {
            "classification": "logistic_regression",
            "regression": "linear_regression",
            "clustering": "kmeans"
        }
        return defaults.get(task_type, "logistic_regression")
    def _get_feature_importance(self, model, feature_names) -> Optional[List[List[Any]]]:
        """Extract feature importance from model"""
        try:
            importance_dict = {}
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
                else:
                    importance_dict = dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
            if importance_dict:
                # Sort by importance and return top 10
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                return sorted_importance[:10]
            return None
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            return None
class AsyncMLService:
    def __init__(self):
        self.ml_service = MLService()
        self.executor = ThreadPoolExecutor(max_workers=4)
    async def train_model_async(
        self,
        file_path: str,
        task_type: str,
        model_type: Optional[str] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train model asynchronously in background thread"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.ml_service.train_model,
                file_path,
                task_type,
                model_type,
                target_column
            )
            return result
        except Exception as e:
            logger.error(f"Async training error: {str(e)}")
            raise ModelTrainingError(f"Async training failed: {str(e)}")
    async def predict_async(self, model_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                model_path,
                data
            )
            return result
        except Exception as e:
            logger.error(f"Async prediction error: {str(e)}")
            raise ModelTrainingError(f"Async prediction failed: {str(e)}")
    def _predict_sync(self, model_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous prediction method"""
        try:
            model = joblib.load(model_path)
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model: {str(e)}")
        # Load metadata sidecar if exists to align features
        meta_path = model_path.with_suffix(".json")
        expected_features = None
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    expected_features = meta.get("features", None)
            except Exception:
                expected_features = None
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        # Align input columns if possible using metadata
        if expected_features:
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df[expected_features]
        # If model has feature_names_in_, try to align (fallback)
        try:
            if hasattr(model, 'feature_names_in_'):
                expected = list(model.feature_names_in_)
                for col in expected:
                    if col not in input_df.columns:
                        input_df[col] = np.nan
                input_df = input_df[expected]
        except Exception:
            pass
        # Make prediction
        try:
            prediction = model.predict(input_df)
        except Exception as e:
            raise ModelTrainingError(f"Model prediction failed: {str(e)}")
        # Get prediction probabilities if available
        probabilities = None
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df).tolist()
        except Exception:
            probabilities = None
        pred_list = prediction.tolist() if hasattr(prediction, 'tolist') else [prediction]
        return {
            "prediction": pred_list,
            "probabilities": probabilities,
            "model_id": model_path.stem
        }