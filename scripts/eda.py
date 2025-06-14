import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from datetime import datetime
import logging
import json
import base64
from io import BytesIO
from typing import Optional, List, Dict, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("EDAAnalyzer")


class EDAAnalyzer:
    """
    Production-grade EDA analyzer implementing industry-standard practices with flexible configuration.

    Features:
    - Configurable analysis methods (outlier detection, missing value handling, etc.)
    - Multiple output formats (static files, notebook display, web-ready formats)
    - Comprehensive data quality assessment
    - Automated reporting with actionable insights
    - Visualization for different contexts (Jupyter, web, static)

    Usage:
    analyzer = EDAAnalyzer(
        df=df,
        target_col='price',
        report_dir='reports',
        output_formats=['html', 'json'],
        visualization_context='notebook'
    )
    analyzer.full_analysis()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        report_dir: str = "eda_reports",
        clean_data_dir: str = "cleaned_data",
        output_formats: List[str] = ["html", "json"],
        visualization_context: str = "static",
        outlier_method: str = "IQR",
        outlier_params: Dict = {"threshold": 1.5},
        missing_value_strategy: str = "auto",
        skewness_threshold: float = 0.5,
        correlation_method: str = "spearman",
    ):
        """
        Initialize the EDA analyzer with configurable parameters.

        Parameters:
        df -- Input DataFrame
        target_col -- Target variable for supervised analysis (optional)
        report_dir -- Directory for saving reports
        clean_data_dir -- Directory for saving cleaned data
        output_formats -- Output formats for reports: ['html', 'json', 'txt']
        visualization_context -- Visualization context: ['static', 'notebook', 'web']
        outlier_method -- Outlier detection method: ['IQR', 'zscore', 'percentile', 'mad']
        outlier_params -- Parameters for outlier detection
        missing_value_strategy -- Strategy for missing values: ['auto', 'drop', 'fill', 'impute']
        skewness_threshold -- Absolute skewness value threshold for transformation
        correlation_method -- Correlation method: ['pearson', 'spearman', 'kendall']
        """
        self.original_df = df.copy(deep=True)
        self.cleaned_df = df.copy(deep=True)
        self.target_col = target_col
        self.report_dir = Path(report_dir)
        self.clean_data_dir = Path(clean_data_dir)
        self.output_formats = output_formats
        self.visualization_context = visualization_context
        self.outlier_method = outlier_method
        self.outlier_params = outlier_params
        self.missing_value_strategy = missing_value_strategy
        self.skewness_threshold = skewness_threshold
        self.correlation_method = correlation_method

        # Initialize directories
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.clean_data_dir.mkdir(parents=True, exist_ok=True)

        # Configure visualization
        self._configure_visualization()

        # Initialize state
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.numeric_cols = []
        self.categorical_cols = []
        self.report_data = {}
        self.analysis_complete = False
        self.plots = {}

        # Classify columns
        self._classify_columns()

        # Configure logging
        self.log_file = self.report_dir / f"{self.timestamp}_eda.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    def _configure_visualization(self):
        """Configure visualization based on context"""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("colorblind")

        if self.visualization_context == "static":
            matplotlib.use("Agg")  # Non-interactive backend
        elif self.visualization_context == "web":
            plt.switch_backend("Agg")  # Non-interactive for web
        elif self.visualization_context == "notebook":
            # Enable inline display for Jupyter
            try:
                from IPython import get_ipython

                ipython = get_ipython()
                if ipython is not None:
                    ipython.run_line_magic("matplotlib", "inline")
            except (ImportError, NameError):
                pass  # Not in notebook environment

    def _classify_columns(self):
        """Classify columns as numeric or categorical"""
        try:
            self.numeric_cols = self.cleaned_df.select_dtypes(
                include=np.number
            ).columns.tolist()
            self.categorical_cols = self.cleaned_df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            if self.target_col:
                if self.target_col in self.numeric_cols:
                    self.numeric_cols.remove(self.target_col)
                if self.target_col in self.categorical_cols:
                    self.categorical_cols.remove(self.target_col)
        except Exception as e:
            logger.error(f"Error classifying columns: {str(e)}")
            raise

    def data_quality_report(self) -> pd.DataFrame:
        """Generate comprehensive data quality report"""
        try:
            dq_report = pd.DataFrame(index=self.cleaned_df.columns)

            dq_report["missing"] = self.cleaned_df.isnull().sum().values
            dq_report["missing_pct"] = (
                dq_report["missing"] / len(self.cleaned_df)
            ).round(3)
            dq_report["dtype"] = self.cleaned_df.dtypes.values
            dq_report["unique"] = self.cleaned_df.nunique().values
            dq_report["duplicates"] = [self.cleaned_df.duplicated().sum()] * len(
                dq_report
            )

            # Add data quality flags
            dq_report["high_missing"] = dq_report["missing_pct"] > 0.3
            dq_report["high_cardinality"] = (dq_report["unique"] > 50) & (
                dq_report["dtype"] == "object"
            )
            dq_report["zero_variance"] = self.cleaned_df.nunique() == 1

            self.report_data["data_quality"] = dq_report
            return dq_report
        except Exception as e:
            logger.error(f"Error generating data quality report: {str(e)}")
            return pd.DataFrame()

    def univariate_analysis(self):
        """Perform univariate analysis with robust statistics"""
        try:
            analysis = {}

            # Numeric features analysis
            if self.numeric_cols:
                num_stats = self.cleaned_df[self.numeric_cols].describe(
                    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
                )
                num_stats.loc["IQR"] = num_stats.loc["75%"] - num_stats.loc["25%"]
                num_stats.loc["skewness"] = self.cleaned_df[self.numeric_cols].skew()
                num_stats.loc["kurtosis"] = self.cleaned_df[
                    self.numeric_cols
                ].kurtosis()
                analysis["numeric"] = num_stats

            # Categorical features analysis
            cat_stats = []
            for col in self.categorical_cols:
                freq = self.cleaned_df[col].value_counts(dropna=False, normalize=True)
                cat_stats.append(freq)
            analysis["categorical"] = cat_stats

            # Visualization
            self._generate_univariate_plots()

            self.report_data["univariate"] = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error in univariate analysis: {str(e)}")
            return {}

    def _generate_univariate_plots(self):
        """Generate univariate visualizations"""
        # Histograms and boxplots for numeric features
        for col in self.numeric_cols:
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Histogram with KDE
                sns.histplot(self.cleaned_df[col], kde=True, ax=ax1)
                ax1.set_title(f"Distribution of {col}")
                ax1.axvline(
                    self.cleaned_df[col].median(),
                    color="r",
                    linestyle="--",
                    label="Median",
                )

                # Boxplot
                sns.boxplot(x=self.cleaned_df[col], ax=ax2)
                ax2.set_title(f"Boxplot of {col}")

                self._handle_plot_output(f"univariate_{col}", fig)
            except Exception as e:
                logger.error(f"Error generating plot for {col}: {str(e)}")

        # Bar plots for categorical features
        for col in self.categorical_cols:
            try:
                fig = plt.figure(figsize=(10, 6))
                order = self.cleaned_df[col].value_counts().index
                sns.countplot(data=self.cleaned_df, x=col, order=order)
                plt.title(f"Distribution of {col}")
                plt.xticks(rotation=45)
                self._handle_plot_output(f"countplot_{col}", fig)
            except Exception as e:
                logger.error(f"Error generating plot for {col}: {str(e)}")

    def multivariate_analysis(self):
        """Perform multivariate analysis"""
        try:
            analysis = {}

            # Correlation analysis
            if self.numeric_cols:
                corr_cols = self.numeric_cols + (
                    [self.target_col]
                    if self.target_col and self.target_col in self.cleaned_df.columns
                    else []
                )
                if len(corr_cols) > 1:  # Need at least 2 columns for correlation
                    corr_matrix = self.cleaned_df[corr_cols].corr(
                        method=self.correlation_method
                    )
                    analysis["correlation"] = corr_matrix

                    if not corr_matrix.empty:
                        try:
                            plt.figure(figsize=(12, 10))
                            sns.heatmap(
                                corr_matrix,
                                annot=True,
                                fmt=".2f",
                                cmap="coolwarm",
                                mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
                            )
                            plt.title(
                                f"{self.correlation_method.capitalize()} Correlation Matrix"
                            )
                            self._handle_plot_output("correlation_matrix")
                        except Exception as e:
                            logger.error(
                                f"Error generating correlation matrix: {str(e)}"
                            )

            # Target analysis
            if self.target_col and self.target_col in self.cleaned_df.columns:
                # Numeric target
                if self.cleaned_df[self.target_col].dtype in [np.int64, np.float64]:
                    for col in self.categorical_cols:
                        try:
                            plt.figure(figsize=(10, 6))
                            sns.boxplot(data=self.cleaned_df, x=col, y=self.target_col)
                            plt.title(f"{self.target_col} by {col}")
                            plt.xticks(rotation=45)
                            self._handle_plot_output(f"target_{col}")
                        except Exception as e:
                            logger.error(
                                f"Error generating plot for {col} vs target: {str(e)}"
                            )

                # Categorical target
                else:
                    for col in self.numeric_cols:
                        try:
                            plt.figure(figsize=(10, 6))
                            sns.boxplot(data=self.cleaned_df, x=self.target_col, y=col)
                            plt.title(f"{col} by {self.target_col}")
                            self._handle_plot_output(f"{col}_by_target")
                        except Exception as e:
                            logger.error(
                                f"Error generating plot for {col} by target: {str(e)}"
                            )

            # Pairplot for small datasets
            if len(self.numeric_cols) <= 8 and len(self.numeric_cols) > 1:
                try:
                    sample_df = self.cleaned_df.sample(min(500, len(self.cleaned_df)))
                    g = sns.pairplot(
                        sample_df[
                            self.numeric_cols
                            + ([self.target_col] if self.target_col else [])
                        ],
                        diag_kind="kde",
                        corner=True,
                    )
                    g.fig.suptitle("Pairwise Relationships", y=1.02)
                    self._handle_plot_output("pairplot", g.fig)
                except Exception as e:
                    logger.error(f"Error generating pairplot: {str(e)}")

            self.report_data["multivariate"] = analysis
            return analysis
        except Exception as e:
            logger.error(f"Error in multivariate analysis: {str(e)}")
            return {}

    def _handle_plot_output(self, plot_name: str, fig: Optional[plt.Figure] = None):
        """
        Handle plot output based on visualization context

        Args:
            plot_name: Identifier for the plot
            fig: Matplotlib figure object (if None, uses current figure)
        """
        if fig is None:
            fig = plt.gcf()

        # Save static files
        file_path = self.report_dir / f"{self.timestamp}_{plot_name}.png"
        fig.savefig(file_path, bbox_inches="tight")
        logger.info(f"Saved plot: {file_path}")

        # Store for notebook/web contexts
        if self.visualization_context in ["notebook", "web"]:
            # Store as base64 for web or notebook display
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            self.plots[plot_name] = f"data:image/png;base64,{img_str}"

        # Display in notebook
        if self.visualization_context == "notebook":
            try:
                plt.show()
            except:
                pass

        plt.close(fig)

    def detect_outliers(
        self, method: Optional[str] = None, params: Optional[Dict] = None
    ) -> dict:
        """
        Detect outliers using specified method

        Args:
            method: Detection method ['IQR', 'zscore', 'percentile', 'mad']
            params: Parameters for the method

        Returns:
            Dictionary of outlier information
        """
        method = method or self.outlier_method
        params = params or self.outlier_params

        try:
            outliers = {}

            for col in self.numeric_cols:
                try:
                    col_data = self.cleaned_df[col].dropna()

                    if method == "IQR":
                        threshold = params.get("threshold", 1.5)
                        q1 = col_data.quantile(0.25)
                        q3 = col_data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        outlier_mask = (self.cleaned_df[col] < lower_bound) | (
                            self.cleaned_df[col] > upper_bound
                        )
                    elif method == "zscore":
                        threshold = params.get("threshold", 3)
                        z_scores = np.abs(stats.zscore(col_data))
                        mean = col_data.mean()
                        std = col_data.std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        outlier_mask = (self.cleaned_df[col] < lower_bound) | (
                            self.cleaned_df[col] > upper_bound
                        )
                    elif method == "percentile":
                        lower_pct = params.get("lower", 0.05)
                        upper_pct = params.get("upper", 0.95)
                        lower_bound = col_data.quantile(lower_pct)
                        upper_bound = col_data.quantile(upper_pct)
                        outlier_mask = (self.cleaned_df[col] < lower_bound) | (
                            self.cleaned_df[col] > upper_bound
                        )
                    elif method == "mad":
                        threshold = params.get("threshold", 3)
                        median = col_data.median()
                        mad = np.median(np.abs(col_data - median))
                        modified_z = 0.6745 * (col_data - median) / mad
                        lower_bound = median - threshold * mad / 0.6745
                        upper_bound = median + threshold * mad / 0.6745
                        outlier_mask = np.abs(modified_z) > threshold
                    else:
                        logger.warning(f"Unknown outlier method: {method}. Using IQR.")
                        continue

                    outliers[col] = {
                        "method": method,
                        "params": params,
                        "count": outlier_mask.sum(),
                        "percentage": outlier_mask.mean().round(4),
                        "indices": outlier_mask[outlier_mask].index.tolist(),
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    }
                except Exception as e:
                    logger.error(f"Error detecting outliers for {col}: {str(e)}")

            self.report_data["outliers"] = outliers
            return outliers
        except Exception as e:
            logger.error(f"Error in outlier detection: {str(e)}")
            return {}

    def handle_skewness(self, threshold: Optional[float] = None) -> dict:
        """
        Identify and transform skewed features

        Args:
            threshold: Absolute skewness threshold (default: class threshold)

        Returns:
            Dictionary of transformation recommendations
        """
        threshold = threshold or self.skewness_threshold

        try:
            transformations = {}
            numeric_df = self.cleaned_df[self.numeric_cols]
            skewness = numeric_df.skew()
            skewed_cols = skewness[abs(skewness) > threshold].index.tolist()

            for col in skewed_cols:
                try:
                    original_skew = skewness[col]
                    col_data = numeric_df[col]

                    # Apply transformations
                    log_transformed = np.log1p(col_data)
                    sqrt_transformed = np.sqrt(col_data - col_data.min() + 1)

                    boxcox_transformed = None
                    if col_data.min() > 0:
                        boxcox_transformed, _ = stats.boxcox(col_data + 1)

                    # Calculate new skewness
                    log_skew = log_transformed.skew()
                    sqrt_skew = sqrt_transformed.skew()
                    boxcox_skew = (
                        stats.skew(boxcox_transformed)
                        if boxcox_transformed is not None
                        else None
                    )

                    # Find best transformation
                    transforms = {
                        "log": abs(log_skew),
                        "sqrt": abs(sqrt_skew),
                        "boxcox": (
                            abs(boxcox_skew)
                            if boxcox_skew is not None
                            else float("inf")
                        ),
                    }
                    best_transform = min(transforms, key=transforms.get)

                    transformations[col] = {
                        "original_skew": original_skew,
                        "log_skew": log_skew,
                        "sqrt_skew": sqrt_skew,
                        "boxcox_skew": boxcox_skew,
                        "recommended_transform": best_transform,
                    }
                except Exception as e:
                    logger.error(f"Error handling skewness for {col}: {str(e)}")

            self.report_data["skewness"] = transformations
            return transformations
        except Exception as e:
            logger.error(f"Error in skewness handling: {str(e)}")
            return {}

    def clean_data(
        self,
        handle_missing: Optional[str] = None,
        handle_outliers: str = "cap",
        handle_skewness: bool = True,
        skew_threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Clean data based on EDA findings

        Args:
            handle_missing: Strategy ['auto', 'drop', 'fill', 'impute'] (default: class strategy)
            handle_outliers: Strategy ['cap', 'remove', 'ignore']
            handle_skewness: Apply transformations
            skew_threshold: Skewness threshold (default: class threshold)
        """
        handle_missing = handle_missing or self.missing_value_strategy
        skew_threshold = skew_threshold or self.skewness_threshold

        try:
            logger.info("Starting data cleaning process")

            # Handle missing values
            self._handle_missing_values(strategy=handle_missing)

            # Handle outliers
            if handle_outliers != "ignore":
                self._handle_outliers(method=handle_outliers)

            # Handle skewness
            if handle_skewness:
                self._apply_skewness_transformations(skew_threshold)

            # Remove duplicates
            self.cleaned_df = self.cleaned_df.drop_duplicates()
            logger.info(
                f"Removed {len(self.original_df) - len(self.cleaned_df)} duplicates"
            )

            # Reclassify columns after transformations
            self._classify_columns()

            logger.info("Data cleaning completed successfully")
            return self.cleaned_df
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return self.original_df

    def _handle_missing_values(self, strategy: str = "auto"):
        """Handle missing values based on selected strategy"""
        try:
            # Generate report if not available
            if "data_quality" not in self.report_data:
                self.data_quality_report()

            # Handle each column with missing values
            for col in self.cleaned_df.columns:
                if self.cleaned_df[col].isnull().sum() > 0:
                    if strategy == "drop":
                        self.cleaned_df = self.cleaned_df.dropna(subset=[col])
                    elif strategy == "fill":
                        # Fill with specific value (extendable)
                        fill_value = 0 if col in self.numeric_cols else "MISSING"
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                    elif strategy == "impute":
                        # Simple imputation (could be extended to model-based)
                        if col in self.numeric_cols:
                            fill_value = self.cleaned_df[col].median()
                        else:
                            fill_value = self.cleaned_df[col].mode()[0]
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                    else:  # 'auto' strategy
                        # Auto strategy: drop columns with >30% missing, impute others
                        missing_pct = self.report_data["data_quality"].loc[
                            col, "missing_pct"
                        ]
                        if missing_pct > 0.3:
                            self.cleaned_df = self.cleaned_df.drop(columns=[col])
                        else:
                            if col in self.numeric_cols:
                                fill_value = self.cleaned_df[col].median()
                            else:
                                fill_value = self.cleaned_df[col].mode()[0]
                            self.cleaned_df[col] = self.cleaned_df[col].fillna(
                                fill_value
                            )
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")

    def _handle_outliers(self, method: str = "cap"):
        """Handle outliers based on detection results"""
        try:
            if "outliers" not in self.report_data:
                self.detect_outliers()

            outliers = self.report_data["outliers"]

            if method == "remove":
                indices_to_remove = set()
                for col, data in outliers.items():
                    if "indices" in data:
                        indices_to_remove.update(data["indices"])
                self.cleaned_df = self.cleaned_df.drop(list(indices_to_remove))

            elif method == "cap":
                for col, data in outliers.items():
                    if "lower_bound" in data and "upper_bound" in data:
                        self.cleaned_df[col] = self.cleaned_df[col].clip(
                            lower=data["lower_bound"], upper=data["upper_bound"]
                        )
        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")

    def _apply_skewness_transformations(self, skew_threshold: float):
        """Apply skewness transformations to features"""
        try:
            if "skewness" not in self.report_data:
                self.handle_skewness(skew_threshold)

            transformations = self.report_data["skewness"]

            for col, data in transformations.items():
                if "recommended_transform" in data:
                    if data["recommended_transform"] == "log":
                        self.cleaned_df[col] = np.log1p(self.cleaned_df[col])
                    elif data["recommended_transform"] == "sqrt":
                        min_val = self.cleaned_df[col].min()
                        offset = 1 - min_val if min_val < 1 else 0
                        self.cleaned_df[col] = np.sqrt(self.cleaned_df[col] + offset)
                    elif data["recommended_transform"] == "boxcox":
                        self.cleaned_df[col] = stats.boxcox(self.cleaned_df[col] + 1)[0]
        except Exception as e:
            logger.error(f"Error applying skewness transformations: {str(e)}")

    def save_cleaned_data(self, filename: str = None):
        """Save cleaned data with versioning"""
        try:
            if filename is None:
                filename = f"cleaned_data_{self.timestamp}.csv"

            filepath = self.clean_data_dir / filename
            self.cleaned_df.to_csv(filepath, index=False)
            logger.info(f"Saved cleaned data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving cleaned data: {str(e)}")
            return None

    def generate_report(self):
        """Generate comprehensive EDA report in specified formats"""
        try:
            if not self.report_data:
                logger.warning("No analysis data found. Run full_analysis() first")
                return []

            report_paths = []

            # Generate HTML report
            if "html" in self.output_formats:
                report_paths.append(self._generate_html_report())

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

    def _generate_html_report(self):
        """Generate interactive HTML report"""
        try:
            from jinja2 import Template

            # Load template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>EDA Report - {{ timestamp }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1, h2 { color: #2c3e50; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    img { max-width: 100%; height: auto; margin: 10px 0; }
                    .section { margin-bottom: 40px; }
                </style>
            </head>
            <body>
                <h1>Exploratory Data Analysis Report</h1>
                <p>Generated at: {{ timestamp }}</p>
                
                <div class="section">
                    <h2>Data Quality Assessment</h2>
                    {{ dq_table }}
                </div>
                
                <div class="section">
                    <h2>Outlier Analysis</h2>
                    <ul>
                        {% for col, data in outliers.items() %}
                        <li>{{ col }}: {{ data.count }} outliers ({{ (data.percentage * 100)|round(2) }}%)</li>
                        {% endfor %}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    {% for plot_name, plot_data in plots.items() %}
                    <div>
                        <h3>{{ plot_name.replace('_', ' ')|title }}</h3>
                        <img src="{{ plot_data }}" alt="{{ plot_name }}">
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <ul>
                        <li>Handle missing values in high-missing columns</li>
                        <li>Address outliers using capping or removal</li>
                        <li>Apply transformations to skewed features</li>
                        <li>Consider dimensionality reduction for high-cardinality features</li>
                    </ul>
                </div>
            </body>
            </html>
            """

            # Prepare data
            dq_table = (
                self.report_data.get("data_quality", pd.DataFrame()).to_html()
                if "data_quality" in self.report_data
                else ""
            )
            outliers = self.report_data.get("outliers", {})

            # Render template
            template = Template(template_str)
            html_content = template.render(
                timestamp=self.timestamp,
                dq_table=dq_table,
                outliers=outliers,
                plots=self.plots,
            )

            # Save report
            report_path = self.report_dir / f"{self.timestamp}_eda_report.html"
            with open(report_path, "w") as f:
                f.write(html_content)

            logger.info(f"Generated HTML report: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return None

    def _generate_json_report(self):
        """Generate machine-readable JSON report"""
        try:
            report_data = self.report_data.copy()

            # Add metadata
            report_data["metadata"] = {
                "timestamp": self.timestamp,
                "dataset_shape": self.original_df.shape,
                "cleaned_shape": (
                    self.cleaned_df.shape if hasattr(self, "cleaned_df") else None
                ),
            }

            # Save report
            report_path = self.report_dir / f"{self.timestamp}_eda_report.json"
            with open(report_path, "w") as f:
                json.dump(report_data, f, default=str, indent=2)

            logger.info(f"Generated JSON report: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            return None

    def _generate_text_report(self):
        """Generate text-based report"""
        try:
            report_path = self.report_dir / f"{self.timestamp}_eda_report.txt"

            with open(report_path, "w") as f:
                f.write("=" * 80 + "\n")
                f.write(f"COMPREHENSIVE EDA REPORT\n")
                f.write(f"Generated at: {self.timestamp}\n")
                f.write("=" * 80 + "\n\n")

                # Data Quality Summary
                f.write("DATA QUALITY ASSESSMENT\n")
                f.write("-" * 60 + "\n")
                dq = self.report_data.get("data_quality", pd.DataFrame())
                if not dq.empty:
                    f.write(dq.to_string())
                    f.write("\n\nKey Issues:\n")
                    f.write(
                        f"- Columns with >30% missing values: {dq[dq['high_missing']].index.tolist()}\n"
                    )
                    f.write(
                        f"- High cardinality categoricals: {dq[dq['high_cardinality']].index.tolist()}\n"
                    )
                    f.write(
                        f"- Zero variance features: {dq[dq['zero_variance']].index.tolist()}\n"
                    )
                f.write("\n\n")

                # Outlier Report
                f.write("OUTLIER ANALYSIS\n")
                f.write("-" * 60 + "\n")
                outliers = self.report_data.get("outliers", {})
                for col, data in outliers.items():
                    f.write(
                        f"{col}: {data['count']} outliers ({data['percentage']:.2%}) using {data['method']}\n"
                    )
                f.write("\n\n")

                # Skewness Report
                f.write("SKEWNESS ANALYSIS\n")
                f.write("-" * 60 + "\n")
                skewness = self.report_data.get("skewness", {})
                for col, data in skewness.items():
                    f.write(f"{col}: Original skew = {data['original_skew']:.2f} | ")
                    f.write(f"Recommended: {data['recommended_transform']}\n")

                # Key Recommendations
                f.write("\n\nKEY RECOMMENDATIONS\n")
                f.write("-" * 60 + "\n")
                f.write("1. Data Cleaning:\n")
                f.write("   - Handle missing values using appropriate strategy\n")
                f.write(
                    "   - Address high-cardinality features with encoding/binning\n"
                )
                f.write("   - Remove zero-variance features\n\n")

                f.write("2. Feature Engineering:\n")
                f.write("   - Apply transformations to skewed features\n")
                f.write("   - Consider interaction terms between key variables\n")
                f.write("   - Create time-based features for temporal data\n\n")

                f.write("3. Modeling Preparation:\n")
                f.write("   - Treat outliers based on domain context\n")
                f.write("   - Scale numeric features before modeling\n")
                f.write("   - Handle class imbalance for classification tasks\n")

            logger.info(f"Generated text report: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating text report: {str(e)}")
            return None

    def full_analysis(self):
        """Run complete EDA workflow"""
        logger.info(f"Starting full EDA analysis at {self.timestamp}")

        try:
            self.data_quality_report()
            self.univariate_analysis()
            self.multivariate_analysis()
            self.detect_outliers()
            self.handle_skewness()
            report_paths = self.generate_report()
            self.analysis_complete = True

            logger.info("EDA analysis completed successfully")
            return report_paths
        except Exception as e:
            logger.error(f"Full analysis failed: {str(e)}")
            return None

    def get_web_visualizations(self) -> Dict[str, str]:
        """Get base64 encoded visualizations for web applications"""
        return self.plots
