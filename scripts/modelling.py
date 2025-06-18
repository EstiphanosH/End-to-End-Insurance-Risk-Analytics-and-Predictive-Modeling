import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore the UndefinedMetricWarning for 0 division in classification report if it still occurs
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.metrics')


# Set a visually appealing style for plots
sns.set_theme(style="whitegrid")

class RiskBasedPricingModel:
    """
    A class to handle data preparation, model training (severity and probability),
    risk-based premium calculation, and model interpretation using SHAP.
    Includes enhanced visualization settings.
    """
    def __init__(self, filepath='/content/drive/MyDrive/data/processed/cleaned_data.csv', features=None):
        """
        Initializes the RiskBasedPricingModel.

        Args:
            filepath (str): Path to the cleaned data CSV file.
            features (list): List of feature names to use. Defaults to ['VehicleAge', 'SumInsured'].
        """
        self.filepath = filepath
        self.features = features if features is not None else ['VehicleAge', 'SumInsured']
        self.df = None
        # We'll keep the original split for training, but predict on X_test_c for premium calculation
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = None, None, None, None
        self.X_clf_train, self.X_clf_test, self.y_clf_train, self.y_clf_test = None, None, None, None

        self.severity_model = None
        self.classifier_model = None

        # Predictions will be made on the classification test set (X_clf_test)
        self.severity_preds_on_clf_test = None
        self.claim_prob_on_clf_test = None
        self.df_result = None # Store the results DataFrame

    def load_and_prepare_data(self):
        """
        Loads the dataset and performs basic data preparation.
        """
        try:
            # Specify dtype for column 32 (index 31) to avoid DtypeWarning
            # You might need to adjust the dtype based on your data
            self.df = pd.read_csv(self.filepath, dtype={31: 'object'}) # Example: reading as object
            print("Data loaded successfully.")
        except FileNotFoundError:
            print(f"Error: File not found at {self.filepath}")
            self.df = None # Indicate that data prep failed
            return
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            self.df = None # Indicate that data prep failed
            return

        # Create HasClaim
        self.df['HasClaim'] = self.df['TotalClaims'] > 0

        # Calculate VehicleAge
        self.df['VehicleAge'] = 2025 - self.df['RegistrationYear']

        # Drop rows with null in important columns
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['TotalPremium', 'CalculatedPremiumPerTerm', 'TotalClaims'])
        rows_after_drop = len(self.df)
        if initial_rows - rows_after_drop > 0:
            print(f"Dropped {initial_rows - rows_after_drop} rows with nulls in key columns.")

        # Ensure features exist in the dataframe
        missing_features = [f for f in self.features if f not in self.df.columns]
        if missing_features:
            print(f"Error: Missing features in data: {missing_features}")
            self.df = None # Indicate that data prep failed
            return

        # Subset for regression (only with claims)
        regression_df = self.df[self.df['HasClaim']].copy()
        X_reg = regression_df[self.features]
        y_reg = regression_df['TotalClaims']
        print(f"Prepared data for severity model ({len(regression_df)} rows with claims).")

        # For classification (claim vs no claim) - Use the entire dataset for splitting
        X_clf = self.df[self.features]
        y_clf = self.df['HasClaim']
        print(f"Prepared data for probability model ({len(self.df)} rows).")

        # Split data for training and testing
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        self.X_clf_train, self.X_clf_test, self.y_clf_train, self.y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

        # Check if test sets are not empty
        if self.X_reg_test.empty or self.X_clf_test.empty:
            print("Error: Test sets are empty after splitting. Check data or test_size.")
            self.df = None # Indicate a problem with data split


    def train_severity_model(self):
        """
        Trains the XGBoost Regressor model for claim severity prediction.
        """
        if self.X_reg_train is None or self.y_reg_train is None or self.X_reg_train.empty:
            print("Severity training data not prepared or is empty. Skipping severity model training.")
            return

        print("\nTraining Claim Severity Model...")
        try:
            self.severity_model = XGBRegressor(random_state=42)
            self.severity_model.fit(self.X_reg_train, self.y_reg_train)

            # Evaluate on the severity test set
            severity_train_preds = self.severity_model.predict(self.X_reg_train)
            severity_test_preds = self.severity_model.predict(self.X_reg_test)

            rmse_train = np.sqrt(mean_squared_error(self.y_reg_train, severity_train_preds))
            r2_train = r2_score(self.y_reg_train, severity_train_preds)
            rmse_test = np.sqrt(mean_squared_error(self.y_reg_test, severity_test_preds))
            r2_test = r2_score(self.y_reg_test, severity_test_preds)


            print(f"Claim Severity Model (Train) -> RMSE: {rmse_train:.2f}, R²: {r2_train:.2f}")
            print(f"Claim Severity Model (Test) -> RMSE: {rmse_test:.2f}, R²: {r2_test:.2f}")


        except Exception as e:
            print(f"An error occurred during severity model training: {e}")
            self.severity_model = None # Indicate that training failed

    def train_claim_classifier(self):
        """
        Trains the XGBoost Classifier model for claim probability prediction.
        Handles class imbalance.
        """
        if self.X_clf_train is None or self.y_clf_train is None or self.X_clf_train.empty:
            print("Classification training data not prepared or is empty. Skipping claim classifier training.")
            return

        print("\nTraining Claim Probability Model...")
        try:
            # Calculate scale_pos_weight for imbalanced data
            neg_count = self.y_clf_train.value_counts()[False]
            pos_count = self.y_clf_train.value_counts()[True]
            scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1

            print(f"Using scale_pos_weight: {scale_pos_weight_value:.2f} due to imbalanced data.")

            self.classifier_model = XGBClassifier(
                eval_metric='logloss',
                random_state=42,
                scale_pos_weight=scale_pos_weight_value # Address class imbalance
            )
            self.classifier_model.fit(self.X_clf_train, self.y_clf_train)

            # Evaluate on the classification test set
            classifier_train_preds = self.classifier_model.predict(self.X_clf_train)
            classifier_test_preds = self.classifier_model.predict(self.X_clf_test)
            claim_prob_train = self.classifier_model.predict_proba(self.X_clf_train)[:, 1]
            claim_prob_test = self.classifier_model.predict_proba(self.X_clf_test)[:, 1]


            print("\nClassification Report (Train):\n", classification_report(self.y_clf_train, classifier_train_preds))
            print("ROC AUC (Train):", roc_auc_score(self.y_clf_train, claim_prob_train))

            print("\nClassification Report (Test):\n", classification_report(self.y_clf_test, classifier_test_preds))
            print("ROC AUC (Test):", roc_auc_score(self.y_clf_test, claim_prob_test))


        except Exception as e:
            print(f"An error occurred during classification model training: {e}")
            self.classifier_model = None # Indicate that training failed


    def calculate_risk_based_premium(self):
        """
        Calculates the risk-based premium using the trained models' predictions
        on the classification test set.

        Returns:
            pd.DataFrame: DataFrame with 'PredictedPremium' and 'ActualPremium'.
        """
        if self.severity_model is None or self.classifier_model is None or self.X_clf_test is None or self.X_clf_test.empty:
            print("Models not trained or classification test data not available/is empty. Skipping premium calculation.")
            self.df_result = None # Ensure df_result is None if calculation fails
            return None

        print("\nCalculating Risk-Based Premium...")
        try:
            # Make predictions on the classification test set (which represents all policies)
            self.claim_prob_on_clf_test = self.classifier_model.predict_proba(self.X_clf_test)[:, 1]
            # For severity, we need to predict for ALL policies in the classification test set
            # even those without claims. The model will predict a value, which represents
            # the expected severity *if a claim were to occur* based on the features.
            self.severity_preds_on_clf_test = self.severity_model.predict(self.X_clf_test)


            # Ensure lengths match before calculation
            if len(self.claim_prob_on_clf_test) != len(self.severity_preds_on_clf_test) or len(self.claim_prob_on_clf_test) != len(self.X_clf_test):
                 print("Error: Mismatch in lengths of predictions or test data for premium calculation. Skipping premium calculation.")
                 self.df_result = None
                 return None

            # Calculate premium using predictions on the same test set
            premium = self.claim_prob_on_clf_test * self.severity_preds_on_clf_test + 1000 + 0.2 * self.severity_preds_on_clf_test

            self.df_result = self.X_clf_test.copy() # Use the classification test features as the base
            self.df_result['PredictedPremium'] = premium
            # Ensure ActualPremium is from the original df using the classification test index
            self.df_result['ActualPremium'] = self.df.loc[self.X_clf_test.index, 'CalculatedPremiumPerTerm'].values


            print("Risk-Based Premium calculated.")
            return self.df_result
        except Exception as e:
            print(f"An error occurred during premium calculation: {e}")
            self.df_result = None
            return None


    def visualize_premium_distribution(self):
        """
        Visualizes the distribution of predicted and actual premiums with enhanced settings.
        Uses the stored df_result.
        Includes corrected legend and options for frequency range modification.
        """
        if self.df_result is None or self.df_result.empty:
            print("Premium results not available or empty for visualization. Skipping visualization.")
            return

        print("\nGenerating Premium Distribution Plot...")
        try:
            plt.figure(figsize=(10, 6)) # Set figure size for better readability

            # Use seaborn's histplot directly, and handle legend separately
            # `hue` creates separate distributions based on the column names
            sns.histplot(data=self.df_result, x='PredictedPremium', color='skyblue', kde=True, label='Predicted Premium', stat='density', common_norm=False)
            sns.histplot(data=self.df_result, x='ActualPremium', color='lightcoral', kde=True, label='Actual Premium', stat='density', common_norm=False)

            # Alternatively, for dodged bars without KDE:
            # sns.histplot(data=self.df_result[['PredictedPremium', 'ActualPremium']].melt(var_name='PremiumType', value_name='PremiumValue'),
            #              x='PremiumValue', hue='PremiumType', kde=False, multiple="dodge", shrink=.8)


            plt.title("Distribution of Predicted vs Actual Premiums", fontsize=14) # Improved title
            plt.xlabel("Premium Value", fontsize=12) # Label axes
            plt.ylabel("Density", fontsize=12) # Label axes (using Density with stat='density')

            # Explicitly add the legend using the labels provided in the histplot calls
            plt.legend(title="Premium Type")

            # --- Modifying the "Range of Frequency" (Approaches) ---
            # You likely mean either:
            # 1. Changing the number of bins (how the data is grouped):
            # You can add the 'bins' parameter to sns.histplot.
            # Example: bins=30 (more bins) or bins=10 (fewer bins)
            # sns.histplot(..., bins=30, ...)

            # 2. Setting the y-axis limit (zooming in/out on frequency):
            # plt.ylim(0, 0.00005) # Example: set y-axis limits from 0 to 0.00005


            plt.tight_layout() # Adjust layout
            plt.show()

        except Exception as e:
            print(f"An error occurred during premium distribution visualization: {e}")


    def interpret_models_with_shap(self):
        """
        Generates SHAP summary plots for both models with enhanced settings.
        Uses the classification test set for interpretation data.
        """
        if self.severity_model is None or self.classifier_model is None or self.X_clf_test is None or self.X_clf_test.empty:
             print("Models not trained or classification test data not available/is empty for SHAP. Skipping SHAP interpretation.")
             return

        print("\n--- SHAP Interpretation for Claim Severity Model ---")
        try:
            # Use X_clf_test for interpreting the severity model as well
            # This shows feature impact on expected severity for *all* policies
            explainer_s = shap.Explainer(self.severity_model, self.X_clf_test) # Corrected line
            shap_values_s = explainer_s(self.X_clf_test) # Corrected line
            plt.figure(figsize=(10, 6)) # Set figure size
            shap.summary_plot(shap_values_s, self.X_clf_test, title="SHAP Summary Plot for Claim Severity Prediction", show=False) # show=False allows custom plot adjustments
            plt.title("SHAP Summary Plot for Claim Severity Prediction", fontsize=14) # Custom title
            plt.xlabel("SHAP value (impact on model output)", fontsize=12) # Improve x-axis label
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred during SHAP interpretation for severity model: {e}")


        print("\n--- SHAP Interpretation for Claim Probability Model ---")
        try:
            explainer_c = shap.Explainer(self.classifier_model, self.X_clf_test) # Use X_clf_test for consistency
            shap_values_c = explainer_c(self.X_clf_test) # Use X_clf_test for consistency
            plt.figure(figsize=(10, 6)) # Set figure size
            shap.summary_plot(shap_values_c, self.X_clf_test, title="SHAP Summary Plot for Claim Probability Prediction", show=False) # show=False
            plt.title("SHAP Summary Plot for Claim Probability Prediction", fontsize=14) # Custom title
            plt.xlabel("SHAP value (impact on model output)", fontsize=12) # Improve x-axis label
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred during SHAP interpretation for probability model: {e}")