import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import json

logger = logging.getLogger(__name__)


class HypothesisTester:
    def __init__(
        self,
        df: pd.DataFrame,
        report_dir: str = "../reports",
        output_formats: Optional[List[str]] = None,
        alpha: float = 0.05,
        visualization_context: str = "notebook",
    ):
        """
        Initialize the hypothesis tester with a dataframe and configuration.
        
        Args:
            df: Input dataframe with data to analyze
            report_dir: Directory to save reports
            output_formats: List of output formats (json, txt, etc.)
            alpha: Significance level for hypothesis tests
            visualization_context: Context for visualization (notebook or file)
        """
        self.df = df
        self.report_dir = Path(report_dir)
        self.output_formats = output_formats or ["json", "txt"]
        self.alpha = alpha
        self.visualization_context = visualization_context
        
        self.report_data = {}
        self.test_results = {}
        
        # Create report directory if it doesn't exist
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def t_test(
        self, 
        variable: str, 
        group_variable: str, 
        equal_var: bool = True
    ) -> Dict:
        """
        Perform t-test to compare means between two groups.
        
        Args:
            variable: Numeric variable to test
            group_variable: Categorical variable with two groups
            equal_var: Whether to assume equal variance
            
        Returns:
            Dictionary with test results
        """
        # Validate inputs
        if variable not in self.df.columns:
            raise ValueError(f"Variable '{variable}' not found in dataframe")
        if group_variable not in self.df.columns:
            raise ValueError(f"Group variable '{group_variable}' not found in dataframe")
            
        # Get unique groups
        groups = self.df[group_variable].unique()
        if len(groups) != 2:
            raise ValueError(f"T-test requires exactly 2 groups, found {len(groups)}")
        
        # Extract data for each group
        group1_data = self.df[self.df[group_variable] == groups[0]][variable].dropna()
        group2_data = self.df[self.df[group_variable] == groups[1]][variable].dropna()
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
        
        # Store results
        result = {
            "test_type": "t_test",
            "variable": variable,
            "group_variable": group_variable,
            "groups": groups.tolist(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "group1_mean": group1_data.mean(),
            "group2_mean": group2_data.mean(),
            "group1_std": group1_data.std(),
            "group2_std": group2_data.std(),
            "group1_count": len(group1_data),
            "group2_count": len(group2_data),
            "equal_var": equal_var,
            "alpha": self.alpha
        }
        
        # Add to test results
        test_key = f"t_test_{variable}_by_{group_variable}"
        self.test_results[test_key] = result
        
        return result
    
    def anova_test(
        self, 
        variable: str, 
        group_variable: str
    ) -> Dict:
        """
        Perform one-way ANOVA test to compare means across multiple groups.
        
        Args:
            variable: Numeric variable to test
            group_variable: Categorical variable with multiple groups
            
        Returns:
            Dictionary with test results
        """
        # Validate inputs
        if variable not in self.df.columns:
            raise ValueError(f"Variable '{variable}' not found in dataframe")
        if group_variable not in self.df.columns:
            raise ValueError(f"Group variable '{group_variable}' not found in dataframe")
            
        # Get unique groups
        groups = self.df[group_variable].unique()
        if len(groups) < 3:
            logger.warning(f"ANOVA typically used for 3+ groups, found {len(groups)}. Consider t-test instead.")
        
        # Create formula and fit model
        formula = f"{variable} ~ C({group_variable})"
        model = ols(formula, data=self.df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Perform Tukey's HSD post-hoc test
        tukey = pairwise_tukeyhsd(
            endog=self.df[variable].dropna(),
            groups=self.df[group_variable].dropna(),
            alpha=self.alpha
        )
        
        # Store results
        result = {
            "test_type": "anova",
            "variable": variable,
            "group_variable": group_variable,
            "groups": groups.tolist(),
            "f_statistic": anova_table["F"][0],
            "p_value": anova_table["PR(>F)"][0],
            "significant": anova_table["PR(>F)"][0] < self.alpha,
            "df_between": anova_table["df"][0],
            "df_within": anova_table["df"][1],
            "alpha": self.alpha,
            "group_means": self.df.groupby(group_variable)[variable].mean().to_dict(),
            "group_counts": self.df.groupby(group_variable)[variable].count().to_dict(),
            "tukey_results": {
                "group1": tukey.data[0].tolist(),
                "group2": tukey.data[1].tolist(),
                "meandiff": tukey.data[2].tolist(),
                "p_value": tukey.data[3].tolist(),
                "lower": tukey.data[4].tolist(),
                "upper": tukey.data[5].tolist(),
                "reject": tukey.data[6].tolist()
            }
        }
        
        # Add to test results
        test_key = f"anova_{variable}_by_{group_variable}"
        self.test_results[test_key] = result
        
        return result
    
    def chi_square_test(
        self, 
        variable1: str, 
        variable2: str
    ) -> Dict:
        """
        Perform chi-square test of independence between two categorical variables.
        
        Args:
            variable1: First categorical variable
            variable2: Second categorical variable
            
        Returns:
            Dictionary with test results
        """
        # Validate inputs
        if variable1 not in self.df.columns:
            raise ValueError(f"Variable '{variable1}' not found in dataframe")
        if variable2 not in self.df.columns:
            raise ValueError(f"Variable '{variable2}' not found in dataframe")
        
        # Create contingency table
        contingency_table = pd.crosstab(self.df[variable1], self.df[variable2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Store results
        result = {
            "test_type": "chi_square",
            "variable1": variable1,
            "variable2": variable2,
            "chi2_statistic": chi2,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "degrees_of_freedom": dof,
            "contingency_table": contingency_table.to_dict(),
            "expected_frequencies": pd.DataFrame(
                expected, 
                index=contingency_table.index, 
                columns=contingency_table.columns
            ).to_dict(),
            "alpha": self.alpha
        }
        
        # Add to test results
        test_key = f"chi_square_{variable1}_vs_{variable2}"
        self.test_results[test_key] = result
        
        return result
    
    def correlation_test(
        self, 
        variable1: str, 
        variable2: str, 
        method: str = "pearson"
    ) -> Dict:
        """
        Perform correlation test between two numeric variables.
        
        Args:
            variable1: First numeric variable
            variable2: Second numeric variable
            method: Correlation method (pearson, spearman, or kendall)
            
        Returns:
            Dictionary with test results
        """
        # Validate inputs
        if variable1 not in self.df.columns:
            raise ValueError(f"Variable '{variable1}' not found in dataframe")
        if variable2 not in self.df.columns:
            raise ValueError(f"Variable '{variable2}' not found in dataframe")
        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError(f"Method must be one of: pearson, spearman, kendall. Got: {method}")
        
        # Extract data
        data1 = self.df[variable1].dropna()
        data2 = self.df[variable2].dropna()
        
        # Get common indices
        common_idx = data1.index.intersection(data2.index)
        data1 = data1.loc[common_idx]
        data2 = data2.loc[common_idx]
        
        # Perform correlation test
        if method == "pearson":
            corr, p_value = stats.pearsonr(data1, data2)
        elif method == "spearman":
            corr, p_value = stats.spearmanr(data1, data2)
        else:  # kendall
            corr, p_value = stats.kendalltau(data1, data2)
        
        # Store results
        result = {
            "test_type": "correlation",
            "variable1": variable1,
            "variable2": variable2,
            "method": method,
            "correlation": corr,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "sample_size": len(data1),
            "alpha": self.alpha
        }
        
        # Add to test results
        test_key = f"correlation_{method}_{variable1}_vs_{variable2}"
        self.test_results[test_key] = result
        
        return result
    
    def plot_group_comparison(
        self, 
        variable: str, 
        group_variable: str, 
        plot_type: str = "boxplot",
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of a variable across groups.
        
        Args:
            variable: Numeric variable to compare
            group_variable: Categorical variable defining groups
            plot_type: Type of plot (boxplot, violin, or bar)
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        if plot_type == "boxplot":
            sns.boxplot(x=group_variable, y=variable, data=self.df)
            plt.title(f"Boxplot of {variable} by {group_variable}")
        elif plot_type == "violin":
            sns.violinplot(x=group_variable, y=variable, data=self.df)
            plt.title(f"Violin plot of {variable} by {group_variable}")
        elif plot_type == "bar":
            sns.barplot(x=group_variable, y=variable, data=self.df)
            plt.title(f"Mean of {variable} by {group_variable}")
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        plt.xlabel(group_variable)
        plt.ylabel(variable)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to: {save_path}")
        
        if self.visualization_context == "notebook":
            plt.show()
        
        plt.close()
    
    def plot_correlation(
        self, 
        variable1: str, 
        variable2: str, 
        hue: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Create scatter plot to visualize correlation between two variables.
        
        Args:
            variable1: First numeric variable
            variable2: Second numeric variable
            hue: Optional categorical variable for color coding
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        if hue:
            sns.scatterplot(x=variable1, y=variable2, hue=hue, data=self.df)
        else:
            sns.scatterplot(x=variable1, y=variable2, data=self.df)
            
        # Add regression line
        sns.regplot(x=variable1, y=variable2, data=self.df, scatter=False, line_kws={"color": "red"})
        
        plt.title(f"Correlation between {variable1} and {variable2}")
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to: {save_path}")
        
        if self.visualization_context == "notebook":
            plt.show()
        
        plt.close()
    
    def plot_chi_square_heatmap(
        self, 
        variable1: str, 
        variable2: str, 
        save_path: Optional[str] = None
    ):
        """
        Create heatmap to visualize contingency table for chi-square test.
        
        Args:
            variable1: First categorical variable
            variable2: Second categorical variable
            save_path: Path to save the plot
        """
        # Create contingency table
        contingency_table = pd.crosstab(
            self.df[variable1], 
            self.df[variable2], 
            normalize="all"
        ) * 100  # Convert to percentage
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(contingency_table, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title(f"Contingency Table: {variable1} vs {variable2} (%)")
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to: {save_path}")
        
        if self.visualization_context == "notebook":
            plt.show()
        
        plt.close()
    
    def run_multiple_tests(
        self, 
        tests_config: List[Dict]
    ) -> Dict[str, Dict]:
        """
        Run multiple hypothesis tests based on configuration.
        
        Args:
            tests_config: List of test configurations
            
        Returns:
            Dictionary with all test results
        """
        for test_config in tests_config:
            test_type = test_config.get("test_type")
            
            try:
                if test_type == "t_test":
                    self.t_test(
                        variable=test_config["variable"],
                        group_variable=test_config["group_variable"],
                        equal_var=test_config.get("equal_var", True)
                    )
                elif test_type == "anova":
                    self.anova_test(
                        variable=test_config["variable"],
                        group_variable=test_config["group_variable"]
                    )
                elif test_type == "chi_square":
                    self.chi_square_test(
                        variable1=test_config["variable1"],
                        variable2=test_config["variable2"]
                    )
                elif test_type == "correlation":
                    self.correlation_test(
                        variable1=test_config["variable1"],
                        variable2=test_config["variable2"],
                        method=test_config.get("method", "pearson")
                    )
                else:
                    logger.warning(f"Unsupported test type: {test_type}")
            except Exception as e:
                logger.error(f"Error running {test_type} test: {str(e)}")
        
        return self.test_results
    
    def generate_report(self) -> List[Path]:
        """
        Generate report of all hypothesis test results.
        
        Returns:
            List of paths to generated reports
        """
        if not self.test_results:
            logger.warning("No test results found. Run tests before generating report.")
            return []
        
        report_paths = []
        
        # Generate JSON report
        if "json" in self.output_formats:
            json_path = self.report_dir / "hypothesis_tests.json"
            with open(json_path, "w") as f:
                json.dump(self.test_results, f, indent=2)
            logger.info(f"Generated JSON report: {json_path}")
            report_paths.append(json_path)
        
        # Generate text report
        if "txt" in self.output_formats:
            txt_path = self.report_dir / "hypothesis_tests.txt"
            with open(txt_path, "w") as f:
                f.write("HYPOTHESIS TESTING RESULTS\n")
                f.write("==========================\n\n")
                
                for test_name, result in self.test_results.items():
                    f.write(f"Test: {test_name}\n")
                    f.write("-" * (len(test_name) + 6) + "\n")
                    
                    test_type = result["test_type"]
                    if test_type == "t_test":
                        f.write(f"T-Test: {result['variable']} by {result['group_variable']}\n")
                        f.write(f"Groups: {result['groups']}\n")
                        f.write(f"T-statistic: {result['t_statistic']:.4f}\n")
                        f.write(f"P-value: {result['p_value']:.4f}\n")
                        f.write(f"Significant at alpha={result['alpha']}: {result['significant']}\n")
                        f.write(f"Group means: {result['group1_mean']:.4f} vs {result['group2_mean']:.4f}\n")
                    
                    elif test_type == "anova":
                        f.write(f"ANOVA: {result['variable']} by {result['group_variable']}\n")
                        f.write(f"F-statistic: {result['f_statistic']:.4f}\n")
                        f.write(f"P-value: {result['p_value']:.4f}\n")
                        f.write(f"Significant at alpha={result['alpha']}: {result['significant']}\n")
                        f.write("Group means:\n")
                        for group, mean in result['group_means'].items():
                            f.write(f"  - {group}: {mean:.4f}\n")
                    
                    elif test_type == "chi_square":
                        f.write(f"Chi-Square: {result['variable1']} vs {result['variable2']}\n")
                        f.write(f"Chi-square statistic: {result['chi2_statistic']:.4f}\n")
                        f.write(f"P-value: {result['p_value']:.4f}\n")
                        f.write(f"Degrees of freedom: {result['degrees_of_freedom']}\n")
                        f.write(f"Significant at alpha={result['alpha']}: {result['significant']}\n")
                    
                    elif test_type == "correlation":
                        f.write(f"Correlation ({result['method']}): {result['variable1']} vs {result['variable2']}\n")
                        f.write(f"Correlation coefficient: {result['correlation']:.4f}\n")
                        f.write(f"P-value: {result['p_value']:.4f}\n")
                        f.write(f"Sample size: {result['sample_size']}\n")
                        f.write(f"Significant at alpha={result['alpha']}: {result['significant']}\n")
                    
                    f.write("\n\n")
            
            logger.info(f"Generated text report: {txt_path}")
            report_paths.append(txt_path)
        
        return report_paths
    
    def generate_insights(self) -> List[str]:
        """
        Generate insights based on hypothesis test results.
        
        Returns:
            List of insight statements
        """
        if not self.test_results:
            logger.warning("No test results found. Run tests before generating insights.")
            return []
        
        insights = []
        
        for test_name, result in self.test_results.items():
            test_type = result["test_type"]
            
            if result["significant"]:
                if test_type == "t_test":
                    var = result["variable"]
                    group_var = result["group_variable"]
                    groups = result["groups"]
                    insights.append(
                        f"There is a significant difference in {var} between {groups[0]} and {groups[1]} "
                        f"({group_var}) (p={result['p_value']:.4f})."
                    )
                
                elif test_type == "anova":
                    var = result["variable"]
                    group_var = result["group_variable"]
                    insights.append(
                        f"There are significant differences in {var} across {group_var} groups "
                        f"(F={result['f_statistic']:.2f}, p={result['p_value']:.4f})."
                    )
                    
                    # Add insights from Tukey's test
                    tukey_results = result["tukey_results"]
                    for i, reject in enumerate(tukey_results["reject"]):
                        if reject:
                            group1 = tukey_results["group1"][i]
                            group2 = tukey_results["group2"][i]
                            meandiff = tukey_results["meandiff"][i]
                            insights.append(
                                f"  - {group1} and {group2} differ significantly in {var} "
                                f"(mean diff: {meandiff:.2f})."
                            )
                
                elif test_type == "chi_square":
                    var1 = result["variable1"]
                    var2 = result["variable2"]
                    insights.append(
                        f"There is a significant association between {var1} and {var2} "
                        f"(χ²={result['chi2_statistic']:.2f}, p={result['p_value']:.4f})."
                    )
                
                elif test_type == "correlation":
                    var1 = result["variable1"]
                    var2 = result["variable2"]
                    method = result["method"]
                    corr = result["correlation"]
                    direction = "positive" if corr > 0 else "negative"
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
                    insights.append(
                        f"There is a significant {strength} {direction} correlation between {var1} and {var2} "
                        f"({method}, r={corr:.2f}, p={result['p_value']:.4f})."
                    )
            else:
                if test_type == "t_test":
                    var = result["variable"]
                    group_var = result["group_variable"]
                    groups = result["groups"]
                    insights.append(
                        f"No significant difference in {var} between {groups[0]} and {groups[1]} "
                        f"({group_var}) (p={result['p_value']:.4f})."
                    )
                
                elif test_type == "anova":
                    var = result["variable"]
                    group_var = result["group_variable"]
                    insights.append(
                        f"No significant differences in {var} across {group_var} groups "
                        f"(F={result['f_statistic']:.2f}, p={result['p_value']:.4f})."
                    )
                
                elif test_type == "chi_square":
                    var1 = result["variable1"]
                    var2 = result["variable2"]
                    insights.append(
                        f"No significant association between {var1} and {var2} "
                        f"(χ²={result['chi2_statistic']:.2f}, p={result['p_value']:.4f})."
                    )
                
                elif test_type == "correlation":
                    var1 = result["variable1"]
                    var2 = result["variable2"]
                    method = result["method"]
                    corr = result["correlation"]
                    insights.append(
                        f"No significant correlation between {var1} and {var2} "
                        f"({method}, r={corr:.2f}, p={result['p_value']:.4f})."
                    )
        
        return insights