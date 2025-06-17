import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.hypothesis_tester import HypothesisTester

class TestHypothesisTester(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        self.report_dir = os.path.join(self.test_dir, 'reports')
        
        # Sample DataFrame with varied data characteristics
        self.df = pd.DataFrame({
            'numeric_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'numeric_2': [5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2, 13.3, 14.4],
            'categorical_1': ['A', 'B', 'A', 'B', 'C', 'C', 'A', 'B', 'C', 'D'],
            'categorical_2': ['X', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'Y'],
            'binary': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Initialize tester with test directories
        self.tester = HypothesisTester(
            df=self.df,
            report_dir=self.report_dir,
            output_formats=['json', 'txt'],
            alpha=0.05
        )
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        self.assertEqual(len(self.tester.df), 10)
        self.assertEqual(self.tester.alpha, 0.05)
        self.assertEqual(self.tester.output_formats, ['json', 'txt'])
    
    def test_t_test(self):
        # Test t-test with binary variable
        result = self.tester.t_test('numeric_1', 'binary')
        self.assertIn('t_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertEqual(result['group1_count'] + result['group2_count'], 10)
    
    def test_t_test_validation(self):
        # Test validation for non-existent column
        with self.assertRaises(ValueError):
            self.tester.t_test('non_existent', 'binary')
        
        # Test validation for non-binary group variable
        with self.assertRaises(ValueError):
            self.tester.t_test('numeric_1', 'categorical_1')
    
    def test_anova_test(self):
        # Test ANOVA with categorical variable
        result = self.tester.anova_test('numeric_1', 'categorical_1')
        self.assertIn('f_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIn('tukey_results', result)
        
        # Check group means
        self.assertEqual(len(result['group_means']), len(self.df['categorical_1'].unique()))
    
    def test_chi_square_test(self):
        # Test chi-square test between two categorical variables
        result = self.tester.chi_square_test('categorical_1', 'categorical_2')
        self.assertIn('chi2_statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIn('contingency_table', result)
    
    def test_correlation_test(self):
        # Test Pearson correlation
        result = self.tester.correlation_test('numeric_1', 'numeric_2')
        self.assertIn('correlation', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertEqual(result['method'], 'pearson')
        
        # Test Spearman correlation
        result = self.tester.correlation_test('numeric_1', 'numeric_2', method='spearman')
        self.assertEqual(result['method'], 'spearman')
        
        # Test Kendall correlation
        result = self.tester.correlation_test('numeric_1', 'numeric_2', method='kendall')
        self.assertEqual(result['method'], 'kendall')
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_group_comparison(self, mock_savefig):
        # Test boxplot
        self.tester.plot_group_comparison(
            'numeric_1', 'categorical_1', plot_type='boxplot',
            save_path=os.path.join(self.report_dir, 'boxplot.png')
        )
        mock_savefig.assert_called_once()
        
        # Reset mock
        mock_savefig.reset_mock()
        
        # Test violin plot
        self.tester.plot_group_comparison(
            'numeric_1', 'categorical_1', plot_type='violin',
            save_path=os.path.join(self.report_dir, 'violin.png')
        )
        mock_savefig.assert_called_once()
        
        # Reset mock
        mock_savefig.reset_mock()
        
        # Test bar plot
        self.tester.plot_group_comparison(
            'numeric_1', 'categorical_1', plot_type='bar',
            save_path=os.path.join(self.report_dir, 'bar.png')
        )
        mock_savefig.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_correlation(self, mock_savefig):
        # Test correlation plot
        self.tester.plot_correlation(
            'numeric_1', 'numeric_2',
            save_path=os.path.join(self.report_dir, 'correlation.png')
        )
        mock_savefig.assert_called_once()
        
        # Reset mock
        mock_savefig.reset_mock()
        
        # Test with hue
        self.tester.plot_correlation(
            'numeric_1', 'numeric_2', hue='categorical_1',
            save_path=os.path.join(self.report_dir, 'correlation_hue.png')
        )
        mock_savefig.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_chi_square_heatmap(self, mock_savefig):
        # Test chi-square heatmap
        self.tester.plot_chi_square_heatmap(
            'categorical_1', 'categorical_2',
            save_path=os.path.join(self.report_dir, 'heatmap.png')
        )
        mock_savefig.assert_called_once()
    
    def test_run_multiple_tests(self):
        # Define test configurations
        tests_config = [
            {
                "test_type": "t_test",
                "variable": "numeric_1",
                "group_variable": "binary"
            },
            {
                "test_type": "anova",
                "variable": "numeric_2",
                "group_variable": "categorical_1"
            },
            {
                "test_type": "chi_square",
                "variable1": "categorical_1",
                "variable2": "categorical_2"
            },
            {
                "test_type": "correlation",
                "variable1": "numeric_1",
                "variable2": "numeric_2",
                "method": "pearson"
            }
        ]
        
        # Run multiple tests
        results = self.tester.run_multiple_tests(tests_config)
        
        # Check results
        self.assertEqual(len(results), 4)
        self.assertIn("t_test_numeric_1_by_binary", results)
        self.assertIn("anova_numeric_2_by_categorical_1", results)
        self.assertIn("chi_square_categorical_1_vs_categorical_2", results)
        self.assertIn("correlation_pearson_numeric_1_vs_numeric_2", results)
    
    def test_generate_report(self):
        # Run a test to populate test_results
        self.tester.t_test('numeric_1', 'binary')
        
        # Generate report
        report_paths = self.tester.generate_report()
        
        # Check report files
        self.assertEqual(len(report_paths), 2)
        self.assertTrue(any(str(p).endswith('.json') for p in report_paths))
        self.assertTrue(any(str(p).endswith('.txt') for p in report_paths))
        
        # Check file contents
        json_path = os.path.join(self.report_dir, 'hypothesis_tests.json')
        self.assertTrue(os.path.exists(json_path))
        
        txt_path = os.path.join(self.report_dir, 'hypothesis_tests.txt')
        self.assertTrue(os.path.exists(txt_path))
    
    def test_generate_insights(self):
        # Run tests to populate test_results
        self.tester.t_test('numeric_1', 'binary')
        self.tester.correlation_test('numeric_1', 'numeric_2')
        
        # Generate insights
        insights = self.tester.generate_insights()
        
        # Check insights
        self.assertGreaterEqual(len(insights), 2)  # At least one insight per test
        self.assertTrue(any('numeric_1' in insight for insight in insights))
        self.assertTrue(any('correlation' in insight.lower() for insight in insights))
    
    def test_empty_report(self):
        # Try to generate report without running tests
        report_paths = self.tester.generate_report()
        self.assertEqual(len(report_paths), 0)
        
        # Try to generate insights without running tests
        insights = self.tester.generate_insights()
        self.assertEqual(len(insights), 0)

if __name__ == '__main__':
    unittest.main()
