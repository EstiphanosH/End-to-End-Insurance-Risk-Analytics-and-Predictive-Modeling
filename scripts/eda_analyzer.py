import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import logging
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import json

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        report_dir: str = "../report",
        clean_data_dir: str = "../data/processed",
        output_formats: Optional[List[str]] = None,
        visualization_context: str = "notebook",
        outlier_method: str = "isolation_forest",
        outlier_params: Optional[dict] = None,
        missing_value_strategy: str = "auto",
        skewness_threshold: float = 0.8,
    ):
        self.df = df
        self.target_col = target_col
        self.report_dir = Path(report_dir)
        self.clean_data_dir = Path(clean_data_dir)
        self.output_formats = output_formats or ["json", "txt"]
        self.visualization_context = visualization_context
        self.outlier_method = outlier_method
        self.outlier_params = outlier_params or {
            "contamination": 0.05,
            "n_estimators": 200,
        }
        self.missing_value_strategy = missing_value_strategy
        self.skewness_threshold = skewness_threshold

        self.report_data = {}
        self.cleaned_df = df.copy()

    def summary_statistics(self) -> pd.DataFrame:
        desc = self.df.describe(include="all").T
        self.report_data["summary_statistics"] = desc
        return desc

    def missing_values(self) -> pd.DataFrame:
        missing = self.df.isnull().sum().to_frame("missing_count")
        missing["missing_pct"] = 100 * missing["missing_count"] / len(self.df)
        self.report_data["missing_values"] = missing
        return missing

    def detect_outliers(self) -> pd.Series:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        iso = IsolationForest(**self.outlier_params, random_state=42)
        preds = iso.fit_predict(self.df[num_cols].fillna(0))
        outliers = pd.Series(preds == -1, index=self.df.index)
        self.report_data["outliers"] = outliers
        self.cleaned_df = self.df.loc[~outliers].copy()
        return outliers

    def skewness_analysis(self) -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        skewness = self.df[num_cols].skew().to_frame("skewness")
        skewness["high_skew"] = skewness["skewness"].abs() > self.skewness_threshold
        self.report_data["skewness"] = skewness
        return skewness

    def correlation_matrix(self) -> pd.DataFrame:
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        corr = self.df[num_cols].corr()
        self.report_data["correlation"] = corr
        return corr

    def plot_distributions(self, save_dir: Optional[str] = None):
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{save_dir}/{col}_dist.png")
            plt.close()

    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        corr = self.correlation_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.close()

    def outlier_summary(self) -> dict:
        outliers = self.report_data.get("outliers")
        if outliers is None:
            outliers = self.detect_outliers()
        total_rows = len(self.df)
        outlier_count = int(outliers.sum())
        usable_rows = total_rows - outlier_count
        summary = {
            "total_rows": total_rows,
            "outlier_count": outlier_count,
            "usable_rows": usable_rows,
            "usable_pct": (
                round(100 * usable_rows / total_rows, 2) if total_rows else 0.0
            ),
        }
        self.report_data["outlier_summary"] = summary
        return summary

    def save_cleaned_data(self, filename: str = "cleaned_data.csv"):
        self.clean_data_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.clean_data_dir / filename
        self.cleaned_df.to_csv(save_path, index=False)
        logger.info(f"Cleaned data saved to: {save_path}")
        return save_path

    def full_analysis(self, save_cleaned: bool = False):
        self.summary_statistics()
        self.missing_values()
        self.detect_outliers()
        self.skewness_analysis()
        self.correlation_matrix()
        self.outlier_summary()
        self.plot_distributions(save_dir=str(self.report_dir / "plots"))
        self.plot_correlation_heatmap(
            save_path=str(self.report_dir / "plots/correlation_heatmap.png")
        )
        logger.info("Full EDA analysis completed.")
        if save_cleaned:
            self.save_cleaned_data()

    def generate_report(self):
        """Generate EDA report in supported formats (JSON, TXT)"""
        try:
            if not self.report_data:
                logger.warning("No analysis data found. Run full_analysis() first")
                return []

            report_paths = []

            # Generate JSON report
            if "json" in self.output_formats:
                report_paths.append(self._generate_json_report())

            # Generate text report
            if "txt" in self.output_formats:
                report_paths.append(self._generate_text_report())

            return report_paths
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return []

    def _generate_json_report(self):
        json_path = self.report_dir / "eda_report.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    k: v.to_dict() if isinstance(v, (pd.DataFrame, pd.Series)) else v
                    for k, v in self.report_data.items()
                },
                f,
                indent=2,
            )
        logger.info(f"Generated JSON report: {json_path}")
        return json_path

    def _generate_text_report(self):
        txt_path = self.report_dir / "eda_report.txt"
        with open(txt_path, "w") as f:
            for key, value in self.report_data.items():
                f.write(f"{key.upper()}\n")
                f.write(str(value))
                f.write("\n\n")
        logger.info(f"Generated text report: {txt_path}")
        return txt_path

    def generate_recommendations(self) -> List[str]:
        recommendations = []
        missing = self.report_data.get("missing_values")
        if missing is not None and (missing["missing_pct"] > 0).any():
            recommendations.append(
                "Consider imputing or dropping columns with missing values."
            )
        skewness = self.report_data.get("skewness")
        if skewness is not None and skewness["high_skew"].any():
            recommendations.append(
                "Apply transformations to reduce skewness in highly skewed features."
            )
        outliers = self.report_data.get("outliers")
        if outliers is not None and outliers.sum() > 0:
            recommendations.append("Investigate and handle detected outliers.")
        return recommendations
