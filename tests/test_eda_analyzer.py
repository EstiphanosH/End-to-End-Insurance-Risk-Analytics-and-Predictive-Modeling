import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.eda_analyzer import EDAAnalyzer

class TestEDAAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output
        self.test_dir = tempfile.mkdtemp()
        self.report_dir = os.path.join(self.test_dir, 'report')
        self.data_dir = os.path.join(self.test_dir, 'data/processed')
        
        # Sample DataFrame with varied data characteristics
        self.df = pd.DataFrame({
            'numeric_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000],  # Contains outlier
            'numeric_2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
            'categorical': ['A', 'B', 'A', 'B', 'C', 'C', 'A', 'B', 'C', 'D'],
            'missing_data': [1, None, 3, None, 5, None, 7, None, 9, None]
        })
        
        # Initialize analyzer with test directories
        self.analyzer = EDAAnalyzer(
            df=self.df,
            report_dir=self.report_dir,
            clean_data_dir=self.data_dir,
            output_formats=['json', 'txt'],
            outlier_params={'contamination': 0.1}
        )
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        self.assertEqual(len(self.analyzer.df), 10)
        self.assertEqual(self.analyzer.target_col, None)
        self.assertEqual(self.analyzer.outlier_method, "isolation_forest")
    
    def test_summary_statistics(self):
        summary = self.analyzer.summary_statistics()
        self.assertIn('numeric_1', summary.index)
        self.assertAlmostEqual(summary.loc['numeric_1', 'mean'], 104.5, delta=0.1)
    
    def test_missing_values(self):
        missing = self.analyzer.missing_values()
        self.assertEqual(missing.loc['missing_data', 'missing_count'], 5)
        self.assertEqual(missing.loc['categorical', 'missing_count'], 0)
    
    @patch('scripts.eda_analyzer.IsolationForest')
    def test_detect_outliers(self, mock_forest):
        # Mock IsolationForest to return 2 outliers
        mock_clf = MagicMock()
        mock_clf.fit_predict.return_value = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, -1])
        mock_forest.return_value = mock_clf
        
        outliers = self.analyzer.detect_outliers()
        self.assertEqual(outliers.sum(), 2)
        self.assertEqual(len(self.analyzer.cleaned_df), 8)
    
    def test_skewness_analysis(self):
        skewness = self.analyzer.skewness_analysis()
        self.assertGreater(skewness.loc['numeric_1', 'skewness'], 3.0)
        self.assertTrue(skewness.loc['numeric_1', 'high_skew'])
    
    def test_correlation_matrix(self):
        corr_matrix = self.analyzer.correlation_matrix()
        self.assertEqual(corr_matrix.shape, (2, 2))  # Only 2 numeric cols
        self.assertTrue('numeric_1' in corr_matrix.index)
    
    @patch('matplotlib.pyplot.show')
    @patch('seaborn.histplot')
    def test_plot_distributions(self, mock_histplot, mock_show):
        self.analyzer.plot_distributions(save_dir=self.report_dir)
        self.assertEqual(mock_histplot.call_count, 2)  # Called for each numeric column
        self.assertTrue(os.path.exists(os.path.join(self.report_dir, 'numeric_1_dist.png')))
    
    @patch('matplotlib.pyplot.show')
    @patch('seaborn.heatmap')
    def test_correlation_heatmap(self, mock_heatmap, mock_show):
        save_path = os.path.join(self.report_dir, 'corr_heatmap.png')
        self.analyzer.plot_correlation_heatmap(save_path=save_path)
        mock_heatmap.assert_called_once()
        self.assertTrue(os.path.exists(save_path))
    
    def test_outlier_summary(self):
        with patch.object(self.analyzer, 'detect_outliers') as mock_detect:
            mock_detect.return_value = pd.Series([False]*8 + [True, True])
            summary = self.analyzer.outlier_summary()
            self.assertEqual(summary['outlier_count'], 2)
            self.assertEqual(summary['usable_rows'], 8)
    
    def test_save_cleaned_data(self):
        self.analyzer.cleaned_df = self.df.iloc[:8]  # Simulate cleaned data
        save_path = self.analyzer.save_cleaned_data()
        self.assertTrue(os.path.exists(save_path))
        loaded = pd.read_csv(save_path)
        self.assertEqual(len(loaded), 8)
    
    @patch.object(EDAAnalyzer, 'plot_distributions')
    @patch.object(EDAAnalyzer, 'plot_correlation_heatmap')
    def test_full_analysis(self, mock_heatmap, mock_dist):
        self.analyzer.full_analysis(save_cleaned=True)
        self.assertIn('summary_statistics', self.analyzer.report_data)
        self.assertIn('missing_values', self.analyzer.report_data)
        self.assertIn('correlation', self.analyzer.report_data)
        self.assertTrue(os.path.exists(os.path.join(self.data_dir, 'cleaned_data.csv')))
    
    def test_report_generation(self):
        self.analyzer.full_analysis()
        reports = self.analyzer.generate_report()
        self.assertEqual(len(reports), 2)
        self.assertTrue(any(str(p).endswith('.json') for p in reports))
        self.assertTrue(any(str(p).endswith('.txt') for p in reports))
        self.assertTrue(os.path.exists(os.path.join(self.report_dir, 'eda_report.json')))
    
    def test_recommendations(self):
        self.analyzer.full_analysis()
        recs = self.analyzer.generate_recommendations()
        self.assertGreater(len(recs), 0)
        self.assertIn('missing values', recs[0])
        self.assertIn('skewness', recs[1])
        self.assertIn('outliers', recs[2])
    
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        analyzer = EDAAnalyzer(empty_df)
        with self.assertLogs(level='WARNING'):
            analyzer.full_analysis()
        self.assertEqual(len(analyzer.report_data.get('summary_statistics', [])), 0)

if __name__ == '__main__':
    unittest.main()
