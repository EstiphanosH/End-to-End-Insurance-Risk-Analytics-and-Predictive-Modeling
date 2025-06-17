# End-to-End Insurance Risk Analytics & Predictive Modeling
## Dive into real insurance data to uncover low-risk segments and build smart models that optimize premiums. 
# Table of Contents 
[Table of Contents](#table-of-contents)

[project-structure](#project-structure)

[setup](#Setup)


[Business Objective](#business-objective)

[Motivation](#motivation)

[Data](#data)

[Learning Outcomes](#learning-outcomes)



[Key Dates	7](#key-dates)

[Deliverables and Tasks to be done](#deliverables-and-tasks-to-be-done)

[Task 1:](#task-1:)

[1.1 Git and GitHub](#1.1-git-and-github)

[1.2 Project Planning \- EDA & Stats](#1.2-project-planning---eda-&-stats)

[Task 2:](#task-2:)

[Data Version Control (DVC)](#data-version-control-\(dvc\))

[Task 3:](#task-3:)

[A/B Hypothesis Testing](#a/b-hypothesis-testing)

[Task 4:](#task-4:)

[Statistical Modeling](#statistical-modeling)


## Project Structure
- `src/` - Source code package
- `scripts/` - Standalone scripts for data processing
- `notebooks/` - Jupyter notebooks for analysis
- `app/` - Main application package
- `tests/` - Test cases
- `data/` - Raw and processed data
- `models/` - Trained models
- `config/` - Configuration files
- `reports/` - Generated figures and reports

## Setup
1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies using `pip install -r requirements.txt`.
4. Run tests using `pytest tests/`.

# Business Objective

Your employer **AlphaCare Insurance Solutions (ACIS)** is committed to developing cutting-edge risk and predictive analytics in the area of car insurance planning and marketing in South Africa. You have recently joined the data analytics team as marketing analytics engineer, and your first project is to analyse historical insurance claim data. The objective of your analyses is to help optimise the marketing strategy as well as discover “low-risk” targets for which the premium could be reduced, hence an opportunity to attract new clients. 

In order to deliver the business objectives, you would need to brush up your knowledge and perform analysis in the following areas:

* **Insurance Terminologies**  
  * Read on how insurance works. Check out the key insurance glossary [50 Common Insurance Terms and What They Mean — Cornerstone Insurance Brokers](https://www.cornerstoneinsurancebrokers.com/blog/common-insurance-terms)  
* **A/B Hypothesis Testing**  
  * Read on the benefits of A/B hypothesis testing  
  * Accept or reject the following null hypothesis  
    * There are no risk differences across provinces   
    * There are no risk differences between zipcodes   
    * There are no significant margin (profit) difference between zip codes   
    * There are not significant risk difference between Women and men   
* **Machine Learning & Statistical Modeling**  
  * For each zipcode, fit a linear regression model that predicts the total claims  
  * Develop a machine learning model that predicts optimal premium values given   
    * Sets of features about the car to be insured  
    * Sets of features about the owner   
    * Sets of features about the location of the owner  
    * Other features you find relevant   
  * Report on the explaining power of the important features that influence your model 
 
## Task 1: 

### 1.1 Git and GitHub 

* Tasks:   
  * Create a git repository for the week with a good Readme  
  * Git version control   
  * CI/CD with Github Actions  
* Key Performance Indicators (KPIs):  
  * Dev Environment Setup.  
  * Relevant skill in the area demonstrated.

### 1.2 Project Planning \- EDA & Stats

* Develop a foundational understanding of the data, assess its quality, and uncover initial patterns in risk and profitability  
* **Tasks**:   
  * Data Understanding  
  * Exploratory Data Analysis (EDA)  
  * Guiding Questions:  
    * What is the overall Loss Ratio (TotalClaims / TotalPremium) for the portfolio? How does it vary by Province, VehicleType, and Gender?  
    * What are the distributions of key financial variables? Are there outliers in TotalClaims or CustomValueEstimate that could skew our analysis?  
    * Are there temporal trends? Did the claim frequency or severity change over the 18-month period?  
    * Which vehicle makes/models are associated with the highest and lowest claim amounts?  
  * Statistical thinking  
* **KPIs**:  
  * Proactivity to self-learn \- sharing references.  
  * EDA techniques to understand data and discover insights,  
  * Demonstrating Stats understanding by using suitable statistical distributions and plots to provide evidence for actionable insights gained from EDA.

## Task 2:  

Establish a reproducible and auditable data pipeline using Data Version Control (DVC), a standard practice in regulated industries.

In finance and insurance, we must be able to reproduce any analysis or model result at any time for auditing, regulatory compliance, or debugging. DVC ensures our data inputs are as rigorously version-controlled as our code.

### Data Version Control ([DVC](https://dvc.org/)) 

* Tasks:  
  * Install DVC  
    * `pip install dvc`  
  * Initialize DVC: In your project directory, initialize DVC  
    * `dvc init`  
  * Set Up Local Remote Storage  
    * Create a Storage Directory  
      * `mkdir /path/to/your/local/storage`  
    * Add the Storage as a DVC Remote  
      * `dvc remote add -d localstorage /path/to/your/local/storage`  
  * Add Your Data:   
    * Place your datasets into your project directory and use DVC to track them  
      * `dvc add <data.csv>`  
  * Commit Changes to Version Control  
    * Create different versions of the data.  
      *   
    * Commit the .dvc files (which include information about your data files and their versions) to your Git repository  
  * Push Data to Local Remote  
    * `dvc push`


## Task 3:

Statistically validate or reject key hypotheses about risk drivers, which will form the basis of our new segmentation strategy.

### A/B Hypothesis Testing 

For this analysis, "risk" will be quantified by two metrics: Claim Frequency (proportion of policies with at least one claim) and Claim Severity (the average amount of a claim, given a claim occurred). "Margin" is defined as (TotalPremium \- TotalClaims).

* Accept or reject the following **Null Hypotheses:**   
1. **H₀:**There are no risk differences across provinces   
2. **H₀:**There are no risk differences between zip codes   
3. **H₀:**There are no significant margin (profit) difference between zip codes   
4. **H₀:**There are not significant risk difference between Women and Men  
* Tasks:  
  * Select Metrics  
    * Choose the key performance indicator (KPI) that will measure the impact of the features being tested.  
  * Data Segmentation  
    * **Group A (Control Group)**: Plans without the feature   
    * **Group B (Test Group)**: Plans with the feature.  
    * For features with more than two classes, you may need to select two categories to split the data as Group A and Group B. You must ensure, however, that the two groups you selected do not have significant statistical differences on anything other than the feature you are testing. For example, the client attributes, the auto property, and insurance plan type are statistically equivalent.   
  * Statistical Testing  
    * Conduct appropriate tests such as chi-squared for categorical data or t-tests or z-test for numerical data to evaluate the impact of these features.  
    * Analyze the p-value from the statistical test:  
      * If p\_value \< 0.05 (typical threshold for significance), reject the null hypothesis. This suggests that the feature tested does have a statistically significant effect on the KPI.  
      * If p\_value \>= 0.05, fail to reject the null hypothesis, suggesting that the feature does not have a significant impact on the KPI.  
  * Analyze and Report  
    * Analyze the statistical outcomes to determine if there's evidence to reject the null hypotheses. Document all findings and interpret the results within the context of their impact on business strategy and customer experience.
 
* **Interpretation & Business Recommendation:** For each rejected hypothesis, provide a clear interpretation of the result in business terms. E.g., We reject the null hypothesis for provinces (p \< 0.01). Specifically, Gauteng exhibits a 15% higher loss ratio than the Western Cape, suggesting a regional risk adjustment to our premiums may be warranted.

## Task 4:

Build and evaluate predictive models that form the core of a dynamic, risk-based pricing system.  
**Modeling Goals:**

1. Claim Severity Prediction (Risk Model): For policies that have a claim, build a model to predict the TotalClaims amount. This model estimates the financial liability associated with a policy.  
   Target Variable: TotalClaims (on the subset of data where claims \> 0).  
   Evaluation Metric: Root Mean Squared Error (RMSE) to penalize large prediction errors, and R-squared.  
2. Premium Optimization (Pricing Framework): Develop a machine learning model to predict an appropriate premium. A naive approach is to predict CalculatedPremiumPerTerm, but a more sophisticated, business-driven approach is required.  
   **Advanced Task**: Build a model to predict the probability of a claim occurring (a binary classification problem). The Risk-Based Premium can then be conceptually framed as: Premium \= (Predicted Probability of Claim \* Predicted Claim Severity) \+ Expense Loading \+ Profit Margin.

### Statistical Modeling

* Tasks:  
  * Data Preparation:   
    * Handling Missing Data: Impute or remove missing values based on their nature and the quantity missing.  
    * Feature Engineering: Create new features that might be relevant to TotalPremium and TotalClaims.  
    * Encoding Categorical Data: Convert categorical data into a numeric format using one-hot encoding or label encoding to make it suitable for modeling.  
    * Train-Test Split: Divide the data into a training set (for building the model) and a test set (for validating the model), typically using a 70:30 or 80:20 ratio.  
  * Modeling Techniques  
    * **Linear Regression**  
    * Decision Trees  
    * Random Forests  
    * **Gradient Boosting Machines (GBMs):**  
      *  **XGBoost**  
  * Model Building  
    * Implement Linear Regression, Random Forests, and XGBoost models  
  * Model Evaluation  
    * Evaluate each model using appropriate metrics like accuracy, precision, recall, and F1-score.  
  * Feature Importance Analysis  
    * Analyze which features are most influential in predicting retention.  
  * Use SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to interpret the model's predictions and understand how individual features influence the outcomes.  
  * Report comparison between each model performance.
