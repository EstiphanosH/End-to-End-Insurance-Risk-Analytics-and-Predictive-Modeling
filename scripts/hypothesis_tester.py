# 📄 scripts/hypothesis_tester.py
"""
Hypothesis Testing Module for Insurance Risk Analysis

This module contains the RiskHypothesisTester class which provides methods
to perform various statistical tests on insurance data to identify risk factors.

Key Features:
- Province risk analysis (chi-square test)
- ZIP code risk and margin analysis (chi-square and t-tests)
- Gender-based claim severity analysis (t-test)
- Comprehensive data preparation and validation
- Robust error handling for missing data
"""

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency


class RiskHypothesisTester:
    """
    A class for performing hypothesis tests on insurance risk data.

    Attributes:
        df (pd.DataFrame): The processed insurance data
    """

    def __init__(self, df):
        """
        Initialize the RiskHypothesisTester with a DataFrame.

        Args:
            df (pd.DataFrame): Input insurance data
        """
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        """Prepare data for analysis by creating derived columns."""
        # Create claim indicator
        self.df['HasClaim'] = self.df['TotalClaims'] > 0

        # Create claim severity (only for policies with claims)
        self.df['ClaimSeverity'] = np.where(
            self.df['HasClaim'],
            self.df['TotalClaims'],
            np.nan
        )

        # Calculate policy margin
        self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

        # Calculate loss ratio (with zero-division protection)
        self.df['LossRatio'] = np.divide(
            self.df['TotalClaims'],
            self.df['TotalPremium'],
            out=np.zeros_like(self.df['TotalClaims']),
            where=(self.df['TotalPremium'] != 0)
        )

    def _t_test(self, group_col, metric, groups=('Male', 'Female')):
        """
        Perform Welch's t-test between two groups for a given metric.

        Args:
            group_col (str): Column name for grouping
            metric (str): Numeric metric to test
            groups (tuple): Two group labels to compare

        Returns:
            tuple: (t-statistic, p-value) or (np.nan, np.nan) if insufficient data
        """
        # Validate inputs
        if group_col not in self.df.columns:
            return np.nan, np.nan

        # Extract groups
        group_a = self.df[self.df[group_col] == groups[0]][metric].dropna()
        group_b = self.df[self.df[group_col] == groups[1]][metric].dropna()

        # Check if both groups have enough data
        if len(group_a) < 2 or len(group_b) < 2:
            return np.nan, np.nan

        # Perform Welch's t-test (does not assume equal variance)
        stat, p = ttest_ind(group_a, group_b, equal_var=False)
        return stat, p

    def _chi_square_test(self, group_col, outcome_col='HasClaim', df_subset=None):
        """
        Perform chi-square test of independence.

        Args:
            group_col (str): Grouping column name
            outcome_col (str): Binary outcome column name
            df_subset (pd.DataFrame): Optional subset of data

        Returns:
            tuple: (chi2 statistic, p-value) or (np.nan, np.nan) if insufficient data
        """
        # Use subset if provided
        df_use = df_subset if df_subset is not None else self.df

        # Validate columns
        if group_col not in df_use.columns or outcome_col not in df_use.columns:
            return np.nan, np.nan

        # Create contingency table
        try:
            table = pd.crosstab(df_use[group_col], df_use[outcome_col])

            # Check table validity
            if table.size < 4 or table.sum().sum() == 0:
                return np.nan, np.nan

            # Perform chi-square test
            chi2, p, dof, _ = chi2_contingency(table)
            return chi2, p
        except Exception:
            return np.nan, np.nan

    def test_province_risk(self):
        """Test claim probability across provinces."""
        return self._chi_square_test('Province', 'HasClaim')

    def test_zipcode_risk(self, top_n=2):
        """
        Test claim probability in top ZIP codes.

        Args:
            top_n (int): Number of top ZIP codes to consider

        Returns:
            tuple: (chi2 statistic, p-value)
        """
        # Get top ZIP codes
        zip_counts = self.df['PostalCode'].value_counts()
        if len(zip_counts) < top_n:
            return np.nan, np.nan

        top_zips = zip_counts.index[:top_n]
        sub_df = self.df[self.df['PostalCode'].isin(top_zips)]
        return self._chi_square_test('PostalCode', 'HasClaim', df_subset=sub_df)

    def test_zipcode_margin(self, top_n=2):
        """
        Test margin difference in top ZIP codes.

        Args:
            top_n (int): Number of top ZIP codes to consider

        Returns:
            tuple: (t-statistic, p-value)
        """
        # Get top ZIP codes
        zip_counts = self.df['PostalCode'].value_counts()
        if len(zip_counts) < top_n:
            return np.nan, np.nan

        top_zips = zip_counts.index[:top_n]
        return self._t_test('PostalCode', 'Margin', groups=tuple(top_zips[:2]))

    def test_gender_risk(self):
        """Test claim severity difference between genders."""
        return self._t_test('Gender', 'ClaimSeverity')

    def run_all_tests(self, top_n_zip=2):
        """
        Run all hypothesis tests.

        Args:
            top_n_zip (int): Number of top ZIP codes to consider

        Returns:
            dict: Dictionary of test results
        """
        return {
            "Province Risk": self.test_province_risk(),
            "Zip Risk": self.test_zipcode_risk(top_n=top_n_zip),
            "Zip Margin": self.test_zipcode_margin(top_n=top_n_zip),
            "Gender Risk": self.test_gender_risk(),
        }