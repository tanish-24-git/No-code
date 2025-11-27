# services/insight_service.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import structlog

logger = structlog.get_logger()

class InsightService:
    VALID_TASK_TYPES = ["classification", "regression", "clustering"]

    def generate_insights(self, summary: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        try:
            insights = []
            suggested_task_type = "clustering"
            suggested_target_column = None

            insights.append(f"Dataset contains {summary['rows']} rows and {len(summary['columns'])} columns")

            missing_cols = [col for col, count in summary['missing_values'].items() if count > 0]
            if missing_cols:
                insights.append(f"Missing values detected in {len(missing_cols)} columns")
            else:
                insights.append("No missing values detected in the dataset")

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            insights.append(f"Dataset contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")

            suggested_task_type, suggested_target_column = self._suggest_ml_task(df, numeric_cols, categorical_cols)

            if suggested_target_column:
                unique_count = df[suggested_target_column].nunique()
                if suggested_task_type == "classification":
                    insights.append(f"'{suggested_target_column}' appears suitable for classification with {unique_count} classes")
                elif suggested_task_type == "regression":
                    insights.append(f"'{suggested_target_column}' appears suitable for regression (continuous values)")
            else:
                insights.append("No obvious target variable detected, clustering recommended for exploratory analysis")

            suggested_missing_strategy = self._suggest_missing_strategy(df, summary['missing_values'])

            return {
                "insights": insights,
                "suggested_task_type": suggested_task_type,
                "suggested_target_column": suggested_target_column,
                "suggested_missing_strategy": suggested_missing_strategy
            }

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return {
                "insights": ["Unable to generate detailed insights"],
                "suggested_task_type": "clustering",
                "suggested_target_column": None,
                "suggested_missing_strategy": "mean"
            }

    def _suggest_ml_task(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> tuple:
        for col in categorical_cols:
            unique_values = df[col].nunique()
            if 2 <= unique_values <= 10:
                return "classification", col
        for col in numeric_cols:
            unique_values = df[col].nunique()
            if unique_values > 20 and df[col].std() > 0:
                return "regression", col
        return "clustering", None

    def _suggest_missing_strategy(self, df: pd.DataFrame, missing_values: Dict[str, int]) -> str:
        total_missing = sum(missing_values.values())
        if total_missing == 0:
            return "mean"
        total_rows = len(df)
        missing_percentage = total_missing / (total_rows * len(df.columns))
        if missing_percentage > 0.3:
            return "drop"
        numeric_cols_with_missing = []
        categorical_cols_with_missing = []
        for col, count in missing_values.items():
            if count > 0:
                if df[col].dtype in ['int64', 'float64']:
                    numeric_cols_with_missing.append(col)
                else:
                    categorical_cols_with_missing.append(col)
        if len(categorical_cols_with_missing) > len(numeric_cols_with_missing):
            return "mode"
        if numeric_cols_with_missing:
            try:
                skewness = df[numeric_cols_with_missing].skew().abs().mean()
                if skewness > 1:
                    return "median"
                else:
                    return "mean"
            except:
                return "mean"
        return "mean"
