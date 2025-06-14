import pytest
import pandas as pd
import numpy as np
import os
import shutil
from unittest.mock import patch, MagicMock
from scripts.eda import EDAAnalyzer  # Assuming the class is in eda_analyzer.py


# Test Data Setup
@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing"""
    data = {
        "numeric1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100],
        "numeric2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000],
        "categorical": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "C"],
        "target_num": [100, 200, 150, 250, 300, 350, 400, 450, 500, 550, 1000],
        "target_cat": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y", "Z"],
        "missing_col": [1, 2, 3, None, None, None, 7, 8, 9, 10, 11],
        "constant_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "high_cardinality": [f"cat_{i}" for i in range(100)] + ["cat_extra"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def analyzer(sample_data):
    """Create an EDAAnalyzer instance with sample data"""
    return EDAAnalyzer(
        df=sample_data,
        target_col="target_num",
        report_dir="test_reports",
        clean_data_dir="test_clean",
        visualization_context="static",
    )


# Setup and Teardown
def setup_function(function):
    if os.path.exists("test_reports"):
        shutil.rmtree("test_reports")
    if os.path.exists("test_clean"):
        shutil.rmtree("test_clean")
    os.makedirs("test_reports", exist_ok=True)
    os.makedirs("test_clean", exist_ok=True)


def teardown_function(function):
    if os.path.exists("test_reports"):
        shutil.rmtree("test_reports")
    if os.path.exists("test_clean"):
        shutil.rmtree("test_clean")


# Test Cases
def test_initialization(analyzer, sample_data):
    """Test class initialization and properties"""
    assert analyzer.df.equals(sample_data)
    assert analyzer.target_col == "target_num"
    assert analyzer.report_dir == Path("test_reports")
    assert analyzer.clean_data_dir == Path("test_clean")
    assert analyzer.visualization_context == "static"
    assert "numeric1" in analyzer.numeric_cols
    assert "categorical" in analyzer.categorical_cols
    assert "target_num" not in analyzer.numeric_cols
    assert analyzer.timestamp is not None


def test_column_classification(analyzer):
    """Test column type classification"""
    assert set(analyzer.numeric_cols) == {
        "numeric1",
        "numeric2",
        "missing_col",
        "constant_col",
    }
    assert set(analyzer.categorical_cols) == {"categorical", "high_cardinality"}


def test_data_quality_report(analyzer):
    """Test data quality report generation"""
    report = analyzer.data_quality_report()

    assert isinstance(report, pd.DataFrame)
    assert "missing" in report.columns
    assert "missing_pct" in report.columns
    assert "dtype" in report.columns
    assert "unique" in report.columns

    # Check specific values
    assert report.loc["missing_col", "missing_pct"] == 3 / 11
    assert report.loc["constant_col", "unique"] == 1
    assert report.loc["high_cardinality", "high_cardinality"] is True


@patch("matplotlib.pyplot.savefig")
def test_univariate_analysis(mock_savefig, analyzer):
    """Test univariate analysis and plotting"""
    analysis = analyzer.univariate_analysis()

    # Check numeric stats
    num_stats = analysis["numeric"]
    assert "numeric1" in num_stats.columns
    assert num_stats.loc["mean", "numeric1"] == pytest.approx(14.545, rel=0.01)
    assert num_stats.loc["skewness", "numeric1"] > 2  # Should be highly skewed

    # Check categorical stats
    cat_stats = analysis["categorical"]
    assert len(cat_stats) == len(analyzer.categorical_cols)

    # Ensure plots were saved
    assert mock_savefig.call_count >= len(analyzer.numeric_cols) + len(
        analyzer.categorical_cols
    )


@patch("matplotlib.pyplot.savefig")
def test_multivariate_analysis(mock_savefig, analyzer):
    """Test multivariate analysis"""
    analysis = analyzer.multivariate_analysis()

    # Check correlation matrix
    corr_matrix = analysis["correlation"]
    assert "numeric1" in corr_matrix.index
    assert "target_num" in corr_matrix.index
    assert corr_matrix.loc["numeric1", "target_num"] > 0.8

    # Ensure plots were saved
    assert mock_savefig.call_count > 0


def test_outlier_detection(analyzer):
    """Test outlier detection with different methods"""
    # Test IQR method
    iqr_outliers = analyzer.detect_outliers(method="IQR", params={"threshold": 1.5})
    assert iqr_outliers["numeric1"]["count"] == 1
    assert iqr_outliers["numeric1"]["percentage"] == pytest.approx(1 / 11, rel=0.01)

    # Test percentile method
    perc_outliers = analyzer.detect_outliers(
        method="percentile", params={"lower": 0.05, "upper": 0.95}
    )
    assert perc_outliers["numeric1"]["count"] == 1

    # Test z-score method
    z_outliers = analyzer.detect_outliers(method="zscore", params={"threshold": 3})
    assert z_outliers["numeric1"]["count"] == 1

    # Test MAD method
    mad_outliers = analyzer.detect_outliers(method="mad", params={"threshold": 3})
    assert mad_outliers["numeric1"]["count"] == 1


def test_skewness_handling(analyzer):
    """Test skewness identification and transformation recommendations"""
    transformations = analyzer.handle_skewness(skew_threshold=0.5)

    assert "numeric1" in transformations
    assert transformations["numeric1"]["original_skew"] > 2
    assert "recommended_transform" in transformations["numeric1"]
    assert transformations["numeric1"]["recommended_transform"] in [
        "log",
        "sqrt",
        "boxcox",
    ]


def test_data_cleaning(analyzer, sample_data):
    """Test data cleaning functionality"""
    cleaned_df = analyzer.clean_data(
        handle_missing="auto", handle_outliers="cap", handle_skewness=True
    )

    # Check missing value handling
    assert cleaned_df["missing_col"].isnull().sum() == 0

    # Check outlier capping
    iqr_outliers = analyzer.detect_outliers()
    upper_bound = iqr_outliers["numeric1"]["upper_bound"]
    assert cleaned_df["numeric1"].max() <= upper_bound

    # Check skewness transformation
    assert "log_numeric1" in cleaned_df.columns or "sqrt_numeric1" in cleaned_df.columns

    # Check duplicate removal
    assert cleaned_df.duplicated().sum() == 0

    # Check constant column removal
    assert "constant_col" not in cleaned_df.columns


def test_report_generation(analyzer):
    """Test report generation in multiple formats"""
    analyzer.full_analysis()
    reports = analyzer.generate_report()

    # Check report files exist
    assert len(reports) >= 2  # At least HTML and JSON
    assert os.path.exists("test_reports")

    # Check HTML report
    html_report = f"test_reports/{analyzer.timestamp}_eda_report.html"
    assert os.path.exists(html_report)
    assert os.path.getsize(html_report) > 0

    # Check JSON report
    json_report = f"test_reports/{analyzer.timestamp}_eda_report.json"
    assert os.path.exists(json_report)
    assert os.path.getsize(json_report) > 0

    # Check text report
    txt_report = f"test_reports/{analyzer.timestamp}_eda_report.txt"
    assert os.path.exists(txt_report)
    assert os.path.getsize(txt_report) > 0


def test_cleaned_data_saving(analyzer):
    """Test saving cleaned data"""
    analyzer.clean_data()
    save_path = analyzer.save_cleaned_data("test_clean_data.csv")

    assert save_path == Path("test_clean/test_clean_data.csv")
    assert os.path.exists(save_path)

    # Check file content
    loaded_df = pd.read_csv(save_path)
    assert not loaded_df.empty
    assert "constant_col" not in loaded_df.columns


def test_error_handling(sample_data):
    """Test exception handling mechanisms"""
    # Create problematic data
    bad_data = sample_data.copy()
    bad_data["bad_col"] = ["text"] * 5 + [np.nan] * 6

    analyzer = EDAAnalyzer(df=bad_data, report_dir="test_reports")

    # Force an error in univariate analysis
    with patch("seaborn.histplot", side_effect=Exception("Test error")):
        analysis = analyzer.univariate_analysis()
        assert analysis is not None  # Should still return partial results

    # Force an error in report generation
    with patch("builtins.open", side_effect=Exception("Test error")):
        report_path = analyzer.generate_report()
        assert report_path is None


def test_categorical_target_analysis(sample_data):
    """Test analysis with categorical target variable"""
    analyzer = EDAAnalyzer(
        df=sample_data, target_col="target_cat", report_dir="test_reports"
    )

    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        analyzer.multivariate_analysis()
        assert mock_savefig.call_count > 0


def test_no_target_analysis(sample_data):
    """Test analysis without target variable"""
    analyzer = EDAAnalyzer(df=sample_data, report_dir="test_reports")

    analysis = analyzer.multivariate_analysis()
    assert "correlation" in analysis
    assert "target_num" not in analysis["correlation"]


def test_web_visualization_context(sample_data):
    """Test web visualization context handling"""
    analyzer = EDAAnalyzer(
        df=sample_data, visualization_context="web", report_dir="test_reports"
    )

    analyzer.univariate_analysis()
    visualizations = analyzer.get_web_visualizations()

    assert isinstance(visualizations, dict)
    assert len(visualizations) > 0
    for img_data in visualizations.values():
        assert img_data.startswith("data:image/png;base64")


def test_notebook_visualization_context(sample_data):
    """Test notebook visualization context handling (mocked)"""
    with patch("IPython.display.display") as mock_display:
        analyzer = EDAAnalyzer(
            df=sample_data, visualization_context="notebook", report_dir="test_reports"
        )

        analyzer.univariate_analysis()
        assert mock_display.called


def test_full_workflow(analyzer):
    """Test the complete EDA workflow"""
    report_data = analyzer.full_analysis()

    # Verify all components were executed
    assert "data_quality" in report_data
    assert "univariate" in report_data
    assert "multivariate" in report_data
    assert "outliers" in report_data
    assert "skewness" in report_data

    # Verify cleaning
    cleaned_df = analyzer.clean_data()
    assert len(cleaned_df) <= len(analyzer.original_df)

    # Verify saving
    save_path = analyzer.save_cleaned_data()
    assert os.path.exists(save_path)

    # Verify reports
    reports = analyzer.generate_report()
    assert len(reports) > 0
