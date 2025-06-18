#!/usr/bin/env python3
"""
Insurance Risk Hypothesis Testing Script

This script uses the HypothesisTester class to perform insurance-specific
risk analysis on processed insurance data.

Key Tests:
- Province risk analysis (chi-square test)
- ZIP code risk and margin analysis
- Gender-based claim severity analysis
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from hypothesis_tester import HypothesisTester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_insurance_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare insurance-specific derived columns"""
    df = df.copy()
    
    # Create claim indicator
    df['HasClaim'] = df['TotalClaims'] > 0

    # Create claim severity (only for policies with claims)
    df['ClaimSeverity'] = np.where(
        df['HasClaim'],
        df['TotalClaims'],
        np.nan
    )

    # Calculate policy margin
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']

    # Calculate loss ratio (with zero-division protection)
    df['LossRatio'] = np.divide(
        df['TotalClaims'],
        df['TotalPremium'],
        out=np.zeros_like(df['TotalClaims']),
        where=(df['TotalPremium'] != 0)
    )
    
    return df

def main():
    # Configuration
    DATA_PATH = "../data/processed/insurance_data_processed.csv"
    REPORT_DIR = "../reports/insurance_tests"
    TOP_ZIP_CODES = 5
    
    logger.info("Starting insurance risk hypothesis testing")
    
    # Load and prepare data
    logger.info(f"Loading data from {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded successfully with {len(df)} records")
    except FileNotFoundError:
        logger.error(f"Data file not found at {DATA_PATH}")
        return
    
    # Prepare insurance-specific data
    df = prepare_insurance_data(df)
    logger.info("Prepared insurance-specific features")
    
    # Initialize hypothesis tester
    tester = HypothesisTester(
        df=df,
        report_dir=REPORT_DIR,
        alpha=0.05,
        visualization_context="file"
    )
    
    # Run insurance-specific tests
    logger.info("Running insurance risk hypothesis tests")
    
    # 1. Province risk analysis
    logger.info("Testing province risk (chi-square)")
    province_result = tester.chi_square_test('Province', 'HasClaim')
    
    # 2. ZIP code risk analysis
    logger.info(f"Testing top {TOP_ZIP_CODES} ZIP codes risk")
    zip_counts = df['PostalCode'].value_counts()
    if len(zip_counts) >= TOP_ZIP_CODES:
        top_zips = zip_counts.index[:TOP_ZIP_CODES].tolist()
        zip_df = df[df['PostalCode'].isin(top_zips)]
        zip_tester = HypothesisTester(zip_df, report_dir=REPORT_DIR)
        zip_risk_result = zip_tester.chi_square_test('PostalCode', 'HasClaim')
        zip_tester.plot_group_comparison(
            'LossRatio', 'PostalCode', plot_type='boxplot',
            save_path=Path(REPORT_DIR) / "zip_code_loss_ratio.png"
        )
    else:
        logger.warning(f"Not enough ZIP codes for analysis (requested top {TOP_ZIP_CODES}, found {len(zip_counts)}")
    
    # 3. ZIP code margin analysis
    logger.info(f"Testing top {TOP_ZIP_CODES} ZIP codes margin")
    if len(zip_counts) >= TOP_ZIP_CODES:
        zip_margin_result = zip_tester.anova_test('Margin', 'PostalCode')
        zip_tester.plot_group_comparison(
            'Margin', 'PostalCode', plot_type='bar',
            save_path=Path(REPORT_DIR) / "zip_code_margin.png"
        )
    
    # 4. Gender claim severity analysis
    logger.info("Testing gender claim severity")
    gender_result = tester.t_test('ClaimSeverity', 'Gender')
    tester.plot_group_comparison(
        'ClaimSeverity', 'Gender', plot_type='violin',
        save_path=Path(REPORT_DIR) / "gender_claim_severity.png"
    )
    
    # 5. Correlation between premium and claims
    logger.info("Testing premium-claims correlation")
    correlation_result = tester.correlation_test('TotalPremium', 'TotalClaims', method='spearman')
    tester.plot_correlation(
        'TotalPremium', 'TotalClaims', hue='Province',
        save_path=Path(REPORT_DIR) / "premium_claims_correlation.png"
    )
    
    # Generate report and insights
    logger.info("Generating test reports")
    report_paths = tester.generate_report()
    
    logger.info("Generating insights")
    insights = tester.generate_insights()
    insights_path = Path(REPORT_DIR) / "insights.txt"
    with open(insights_path, "w") as f:
        f.write("INSURANCE RISK ANALYSIS INSIGHTS\n")
        f.write("===============================\n\n")
        for insight in insights:
            f.write(f"- {insight}\n")
    
    logger.info(f"Insurance risk analysis completed. Reports saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()