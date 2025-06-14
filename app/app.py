# Main application file
import streamlit as st
import pandas as pd
from scripts.eda import EDAAnalyzer
import os
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Automated EDA Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stDownloadButton>button { background-color: #008CBA; }
    .report-title { color: #2c3e50; font-size: 2.5rem; }
    .section-header { border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="report-title">Automated Exploratory Data Analysis</h1>', unsafe_allow_html=True)
st.caption("Powered by Practical Statistics for Data Scientists methodologies")

# File uploader in sidebar
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader(
        "Upload your dataset", 
        type=["csv", "xlsx", "parquet"],
        help="Supports CSV, Excel, and Parquet files"
    )
    
    st.subheader("Analysis Parameters")
    target_col = st.selectbox("Target Variable (optional)", [None] + ([] if uploaded_file is None else pd.read_csv(uploaded_file).columns.tolist()))
    
    outlier_method = st.selectbox(
        "Outlier Detection Method",
        ["IQR", "zscore", "percentile", "mad"],
        index=0
    )
    
    outlier_params = {}
    if outlier_method == "IQR":
        outlier_params['threshold'] = st.slider("IQR Threshold", 1.0, 3.0, 1.5, 0.1)
    elif outlier_method == "zscore":
        outlier_params['threshold'] = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.5)
    elif outlier_method == "percentile":
        outlier_params['lower'] = st.slider("Lower Percentile", 0.01, 0.2, 0.05, 0.01)
        outlier_params['upper'] = st.slider("Upper Percentile", 0.8, 0.99, 0.95, 0.01)
    elif outlier_method == "mad":
        outlier_params['threshold'] = st.slider("MAD Threshold", 2.0, 5.0, 3.0, 0.5)
    
    st.subheader("Output Options")
    show_cleaned_data = st.checkbox("Show cleaned data preview", True)
    auto_download = st.checkbox("Automatically download report", True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# Main app functionality
if uploaded_file is not None:
    # Load data
    try:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'parquet':
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()
            
        # Show data preview
        with st.expander("Raw Data Preview", expanded=True):
            st.dataframe(df.head(), use_container_width=True)
            st.caption(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
        # Initialize analyzer
        st.session_state.analyzer = EDAAnalyzer(
            df=df,
            target_col=target_col,
            visualization_context='web',
            output_formats=['html', 'json'],
            outlier_method=outlier_method,
            outlier_params=outlier_params
        )
        
        # Analysis button
        if st.button("Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("Performing comprehensive analysis..."):
                # Run analysis
                st.session_state.analyzer.full_analysis()

Fall in Love with Learning resource, [6/14/2025 3:57 AM]
# Clean data
                cleaned_df = st.session_state.analyzer.clean_data()
                
                # Save cleaned data to session
                st.session_state.cleaned_df = cleaned_df
                st.session_state.report_generated = True
                
            st.success("Analysis completed successfully!")
            
        # Show results if analysis complete
        if st.session_state.report_generated:
            # Display visualizations
            st.markdown('<h2 class="section-header">Key Visualizations</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            visualizations = st.session_state.analyzer.get_web_visualizations()
            
            for plot_name, img_data in visualizations.items():
                # Simplify plot names for display
                display_name = plot_name.replace('_', ' ').title()
                
                if 'univariate' in plot_name:
                    with col1:
                        st.markdown(f"{display_name}")
                        st.image(img_data, use_column_width=True)
                elif 'countplot' in plot_name:
                    with col2:
                        st.markdown(f"{display_name}")
                        st.image(img_data, use_column_width=True)
                elif 'correlation' in plot_name:
                    st.markdown(f"{display_name}")
                    st.image(img_data, use_column_width=True)
                elif 'target' in plot_name or 'by_target' in plot_name:
                    st.markdown(f"{display_name}")
                    st.image(img_data, use_column_width=True)
            
            # Data quality report
            st.markdown('<h2 class="section-header">Data Quality Report</h2>', unsafe_allow_html=True)
            dq_report = st.session_state.analyzer.report_data.get('data_quality', pd.DataFrame())
            if not dq_report.empty:
                st.dataframe(dq_report, use_container_width=True)
                
                # Highlight key issues
                if dq_report['high_missing'].any():
                    high_missing = dq_report[dq_report['high_missing']].index.tolist()
                    st.warning(f"⚠️ Columns with >30% missing values: {', '.join(high_missing)}")
                
                if dq_report['high_cardinality'].any():
                    high_cardinality = dq_report[dq_report['high_cardinality']].index.tolist()
                    st.warning(f"⚠️ High cardinality categoricals: {', '.join(high_cardinality)}")
            
            # Outlier analysis
            st.markdown('<h2 class="section-header">Outlier Analysis</h2>', unsafe_allow_html=True)
            outliers = st.session_state.analyzer.report_data.get('outliers', {})
            if outliers:
                outlier_df = pd.DataFrame.from_dict(outliers, orient='index')
                st.dataframe(outlier_df[['count', 'percentage', 'method']], use_container_width=True)
            
            # Recommendations
            st.markdown('<h2 class="section-header">Actionable Recommendations</h2>', unsafe_allow_html=True)
            st.markdown("""
            1. Data Cleaning:
               - Address missing values using appropriate imputation
               - Handle high-cardinality features with encoding or binning
               - Treat outliers based on domain context
            
            2. Feature Engineering:
               - Apply transformations to skewed features
               - Create interaction features between important variables
               - Generate time-based features for temporal data
            
            3. Modeling Preparation:
               - Scale numerical features before modeling
- Address class imbalance for classification tasks
               - Perform feature selection to reduce dimensionality
            """)
            
            # Download section
            st.markdown('<h2 class="section-header">Download Results</h2>', unsafe_allow_html=True)
            
            # Download cleaned data
            cleaned_csv = st.session_state.cleaned_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=cleaned_csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
            
            # Download reports
            col1, col2 = st.columns(2)
            
            with col1:
                # HTML report download
                with open(st.session_state.analyzer.report_dir / f"{st.session_state.analyzer.timestamp}_eda_report.html", "rb") as f:
                    html_report = f.read()
                
                st.download_button(
                    label="Download HTML Report",
                    data=html_report,
                    file_name="eda_report.html",
                    mime="text/html"
                )
            
            with col2:
                # JSON report download
                with open(st.session_state.analyzer.report_dir / f"{st.session_state.analyzer.timestamp}_eda_report.json", "rb") as f:
                    json_report = f.read()
                
                st.download_button(
                    label="Download JSON Report",
                    data=json_report,
                    file_name="eda_report.json",
                    mime="application/json"
                )
            
            # Show cleaned data preview
            if show_cleaned_data:
                with st.expander("Cleaned Data Preview", expanded=True):
                    st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
                    st.caption(f"Cleaned dataset shape: {st.session_state.cleaned_df.shape[0]} rows, {st.session_state.cleaned_df.shape[1]} columns")
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    # Show welcome state
    st.info("👈 Upload a dataset to begin analysis")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*_dOfsE3cVq7s0BQw8E7tEQ.png", use_column_width=True)
    
    # Features list
    st.markdown("""
    ### Key Features:
    - Automated exploratory data analysis
    - Statistical quality assessment
    - Outlier detection with multiple methods
    - Visualization generation
    - Data cleaning recommendations
    - Report generation (HTML/JSON)
    """)