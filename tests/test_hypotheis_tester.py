import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from scipy.stats import ttest_ind, chi2_contingency
from hypothesis_tester import RiskHypothesisTester

# Sample data generators
def create_sufficient_data():
    """Create dataset with sufficient data for all tests"""
    return pd.DataFrame({
        'TotalClaims': [0, 1000, 0, 2000, 700, 0, 500, 1500, 0, 0],
        'TotalPremium': [1000, 1500, 1200, 1800, 900, 800, 2000, 2500, 750, 850],
        'Province': ['ON', 'QC', 'ON', 'QC', 'BC', 'BC', 'AB', 'AB', 'SK', 'MB'],
        'PostalCode': ['M5V', 'H2X', 'M5V', 'H2X', 'V6C', 'V6C', 'T2P', 'T2P', 'S7K', 'R3B'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
    })

def create_insufficient_data():
    """Create dataset that triggers all insufficient data cases"""
    return pd.DataFrame({
        'TotalClaims': [0, 0, 0, 0],
        'TotalPremium': [1000, 1500, 1200, 1800],
        'Province': ['ON', 'ON', 'ON', 'ON'],
        'PostalCode': ['M5V', 'M5V', 'M5V', 'M5V'],
        'Gender': ['Male', 'Male', 'Female', 'Female']
    })

def create_edge_case_data():
    """Create dataset for edge case testing"""
    return pd.DataFrame({
        'TotalClaims': [0, 0, 0, 1, 10000],
        'TotalPremium': [100, 100, 100, 100, 100],
        'Province': ['ON', 'QC', 'ON', 'QC', 'ON'],
        'PostalCode': ['A', 'B', 'A', 'B', 'C'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male']
    })

# Test class
class TestRiskHypothesisTester:
    # ---- Test Initialization ----
    def test_data_preparation(self):
        """Test derived columns are created correctly"""
        df = create_sufficient_data()
        tester = RiskHypothesisTester(df)
        
        assert 'HasClaim' in tester.df.columns
        assert 'Margin' in tester.df.columns
        assert 'LossRatio' in tester.df.columns
        assert 'ClaimSeverity' in tester.df.columns
        
        # Check claim indicator
        assert tester.df['HasClaim'].sum() == 5
        # Check margin calculation
        assert tester.df.loc[1, 'Margin'] == 1500 - 1000
        # Check loss ratio with zero premium protection
        zero_premium_row = pd.DataFrame({
            'TotalClaims': [100],
            'TotalPremium': [0],
            'Province': ['ON'],
            'PostalCode': ['K1A'],
            'Gender': ['Male']
        })
        tester_zero = RiskHypothesisTester(zero_premium_row)
        assert tester_zero.df.loc[0, 'LossRatio'] == 0

    # ---- Core Functionality Tests ----
    @patch('hypothesis_tester.chi2_contingency')
    def test_province_risk(self, mock_chi2):
        """Test province risk analysis"""
        df = create_sufficient_data()
        tester = RiskHypothesisTester(df)
        
        # Mock chi-square to return known values
        mock_chi2.return_value = (3.84, 0.05, 1, np.array([[5, 5], [5, 5]]))
        chi2, p = tester.test_province_risk()
        
        assert not np.isnan(chi2)
        assert not np.isnan(p)
        mock_chi2.assert_called_once()
        
        # Test with insufficient data
        df_insuf = create_insufficient_data()
        tester_insuf = RiskHypothesisTester(df_insuf)
        chi2_insuf, p_insuf = tester_insuf.test_province_risk()
        assert np.isnan(chi2_insuf)
        assert np.isnan(p_insuf)

    def test_zipcode_risk(self):
        """Test ZIP code risk analysis"""
        df = create_sufficient_data()
        tester = RiskHypothesisTester(df)
        
        # Test with sufficient data
        chi2, p = tester.test_zipcode_risk(top_n=2)
        assert not np.isnan(chi2)
        assert not np.isnan(p)
        
        # Test with insufficient ZIP codes
        chi2_insuf, p_insuf = tester.test_zipcode_risk(top_n=10)
        assert np.isnan(chi2_insuf)
        assert np.isnan(p_insuf)
        
        # Test with single ZIP code
        df_single_zip = df.copy()
        df_single_zip['PostalCode'] = 'M5V'
        tester_single = RiskHypothesisTester(df_single_zip)
        chi2_single, p_single = tester_single.test_zipcode_risk(top_n=2)
        assert np.isnan(chi2_single)
        assert np.isnan(p_single)

    @patch('hypothesis_tester.ttest_ind')
    def test_zipcode_margin(self, mock_ttest):
        """Test ZIP code margin analysis"""
        df = create_sufficient_data()
        tester = RiskHypothesisTester(df)
        
        # Mock t-test to return known values
        mock_ttest.return_value = (2.0, 0.04)
        t, p = tester.test_zipcode_margin(top_n=2)
        
        assert not np.isnan(t)
        assert not np.isnan(p)
        mock_ttest.assert_called_once()
        
        # Test with insufficient data
        df_insuf = create_insufficient_data()
        tester_insuf = RiskHypothesisTester(df_insuf)
        t_insuf, p_insuf = tester_insuf.test_zipcode_margin(top_n=2)
        assert np.isnan(t_insuf)
        assert np.isnan(p_insuf)

    @patch('hypothesis_tester.ttest_ind')
    def test_gender_risk(self, mock_ttest):
        """Test gender-based claim severity analysis"""
        df = create_sufficient_data()
        tester = RiskHypothesisTester(df)
        
        # Mock t-test to return known values
        mock_ttest.return_value = (1.96, 0.05)
        t, p = tester.test_gender_risk()
        
        assert not np.isnan(t)
        assert not np.isnan(p)
        mock_ttest.assert_called_once()
        
        # Test with no claims
        df_no_claims = df.copy()
        df_no_claims['TotalClaims'] = 0
        tester_no_claims = RiskHypothesisTester(df_no_claims)
        t_nc, p_nc = tester_no_claims.test_gender_risk()
        assert np.isnan(t_nc)
        assert np.isnan(p_nc)
        
        # Test with only one gender having claims
        df_male_only = df.copy()
        df_male_only.loc[df_male_only['Gender'] == 'Female', 'TotalClaims'] = 0
        tester_male_only = RiskHypothesisTester(df_male_only)
        t_mo, p_mo = tester_male_only.test_gender_risk()
        assert np.isnan(t_mo)
        assert np.isnan(p_mo)

    # ---- Full Workflow Test ----
    def test_run_all_tests(self):
        """Test full analysis workflow"""
        df = create_sufficient_data()
        tester = RiskHypothesisTester(df)
        
        results = tester.run_all_tests(top_n_zip=2)
        
        assert isinstance(results, dict)
        assert set(results.keys()) == {
            "Province Risk", "Zip Risk", "Zip Margin", "Gender Risk"
        }
        
        # Verify all results are tuples with two values
        for value in results.values():
            assert isinstance(value, tuple)
            assert len(value) == 2
        
        # Test with insufficient data
        df_insuf = create_insufficient_data()
        tester_insuf = RiskHypothesisTester(df_insuf)
        results_insuf = tester_insuf.run_all_tests(top_n_zip=2)
        
        for value in results_insuf.values():
            assert np.isnan(value[0])
            assert np.isnan(value[1])

    # ---- Edge Case Tests ----
    def test_zero_premium_edge_case(self):
        """Test handling of zero premium values"""
        df = pd.DataFrame({
            'TotalClaims': [100, 200, 0],
            'TotalPremium': [0, 100, 100],
            'Province': ['ON', 'QC', 'ON'],
            'PostalCode': ['A', 'B', 'A'],
            'Gender': ['Male', 'Female', 'Male']
        })
        tester = RiskHypothesisTester(df)
        
        # Should handle zero premium without errors
        assert tester.df['LossRatio'].iloc[0] == 0
        assert tester.df['LossRatio'].iloc[1] == 2.0  # 200/100
        
        # Should still run analyses
        province_result = tester.test_province_risk()
        assert not any(np.isnan(v) for v in province_result)

    def test_single_category_edge_case(self):
        """Test single category in grouping variable"""
        df = create_edge_case_data()
        tester = RiskHypothesisTester(df)
        
        # Province test with multiple provinces
        chi2, p = tester.test_province_risk()
        assert not np.isnan(chi2)
        
        # Gender test with only one claim in a group
        t, p = tester.test_gender_risk()
        assert np.isnan(t)  # Should be nan because female group has only one claim

    def test_missing_columns(self):
        """Test handling of missing required columns"""
        df = create_sufficient_data().drop(columns=['Province'])
        tester = RiskHypothesisTester(df)
        
        # Should return nan for province test
        chi2, p = tester.test_province_risk()
        assert np.isnan(chi2)
        assert np.isnan(p)
        
        # Should still run other tests
        zip_result = tester.test_zipcode_risk()
        assert not any(np.isnan(v) for v in zip_result)

# Run tests
if __name__ == "__main__":
    pytest.main(["-v", "test_hypothesis_tester.py"])