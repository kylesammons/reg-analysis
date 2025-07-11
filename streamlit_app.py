import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_extras.metric_cards import style_metric_cards
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Regression Analysis",
    page_icon=":chart_with_upwards_trend:",
    layout="wide"
)

# Initialize session state for prediction calculator
if 'prediction_values' not in st.session_state:
    st.session_state.prediction_values = {}

# Initialize session state for analysis results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .interpretation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .pill-indicator {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 20px;
        font-size: 1.0rem;
        font-weight: 500;
        margin-top: 20px;
        margin-bottom: 24px;
        text-align: center;
        width: 100%;
        box-sizing: border-box;
    }
    .pill-green {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .pill-orange {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .pill-red {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .slider-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample datasets for users to try"""
    datasets = {
        "Marketing Campaign Performance": pd.DataFrame({
            'TV_Spend_Thousands': [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 
                                  45, 80, 110, 140, 165, 190, 215, 240, 265, 285,
                                  60, 85, 115, 135, 160, 185, 210, 235, 260, 290],
            'Website_Traffic_Thousands': [12, 18, 24, 30, 36, 42, 48, 54, 60, 66,
                                         10, 19, 26, 32, 38, 44, 50, 56, 62, 68,
                                         14, 20, 27, 33, 39, 45, 51, 57, 63, 69],
            'Social_Media_Spend_Thousands': [15, 22, 28, 35, 42, 48, 55, 62, 68, 75,
                                           12, 25, 30, 38, 45, 52, 58, 65, 72, 78,
                                           18, 24, 32, 40, 47, 54, 60, 67, 74, 80],
            'Brand_Awareness_Percent': [25, 32, 38, 45, 52, 58, 65, 72, 78, 85,
                                       22, 35, 41, 48, 55, 61, 68, 75, 81, 87,
                                       28, 36, 43, 50, 57, 63, 70, 77, 83, 89],
            'Lead_Generation_Count': [120, 180, 240, 300, 360, 420, 480, 540, 600, 660,
                                     100, 195, 255, 315, 375, 435, 495, 555, 615, 675,
                                     135, 190, 250, 310, 370, 430, 490, 550, 610, 670],
            'Cost_Per_Lead': [45, 42, 38, 35, 32, 29, 26, 23, 20, 17,
                             48, 40, 36, 33, 30, 27, 24, 21, 18, 15,
                             46, 41, 37, 34, 31, 28, 25, 22, 19, 16]
        }),
        "Sales Performance Analysis": pd.DataFrame({
            'Sales_Rep_Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                          1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 11,
                                          0.5, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8],
            'Monthly_Revenue_Thousands': [45, 55, 68, 78, 88, 95, 105, 115, 125, 135,
                                         48, 58, 70, 80, 90, 98, 108, 118, 128, 140,
                                         42, 52, 65, 75, 85, 92, 102, 112, 122, 132],
            'Prospect_Calls_Per_Day': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                                      22, 32, 37, 42, 47, 52, 57, 62, 67, 72,
                                      28, 33, 38, 43, 48, 53, 58, 63, 68, 73],
            'Conversion_Rate_Percent': [8, 12, 15, 18, 22, 25, 28, 32, 35, 38,
                                       6, 14, 17, 20, 24, 27, 30, 34, 37, 40,
                                       10, 16, 19, 22, 26, 29, 32, 36, 39, 42],
            'Deal_Size_Average_Thousands': [12, 15, 18, 22, 25, 28, 32, 35, 38, 42,
                                          10, 16, 19, 23, 26, 29, 33, 36, 39, 44,
                                          14, 17, 20, 24, 27, 30, 34, 37, 40, 45],
            'Customer_Satisfaction_Score': [7.2, 7.8, 8.1, 8.4, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7,
                                           7.0, 8.0, 8.3, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8,
                                           7.5, 8.2, 8.5, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 9.9]
        })
    }
    return datasets

def load_data():
    """Load and validate uploaded data"""
    
    # Sample data option
    use_sample = st.checkbox("🎯 Try with sample data", help="Use built-in sample datasets to explore the tool")
    
    if use_sample:
        datasets = create_sample_data()
        dataset_choice = st.selectbox(
            "Choose a sample dataset:",
            list(datasets.keys()),
            help="Select a sample dataset to practice regression analysis"
        )
        return datasets[dataset_choice]
    
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV, Excel, or TSV)",
        type=['csv', 'xlsx', 'xls', 'tsv'],
        help="Upload a file containing your data for regression analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, delimiter='\t')
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    return None

def get_numeric_columns(df):
    """Get numeric columns from dataframe"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols

def check_data_quality(df, x_col, y_col):
    """Check data quality and provide warnings"""
    warnings = []
    suggestions = []
    
    # Check for missing values
    x_missing = df[x_col].isna().sum()
    y_missing = df[y_col].isna().sum()
    
    if x_missing > 0 or y_missing > 0:
        warnings.append(f"⚠️ Missing values detected: {x_col}: {x_missing}, {y_col}: {y_missing}")
        suggestions.append("Consider removing or imputing missing values")
    
    # Check sample size
    clean_df = df[[x_col, y_col]].dropna()
    n = len(clean_df)
    
    if n < 10:
        warnings.append(f"⚠️ Small sample size: {n} observations")
        suggestions.append("Consider collecting more data for reliable results")
    elif n < 30:
        warnings.append(f"⚠️ Moderate sample size: {n} observations")
        suggestions.append("Results may be more reliable with additional data")
    
    # Check for outliers using IQR method
    def detect_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    x_outliers = detect_outliers(clean_df[x_col])
    y_outliers = detect_outliers(clean_df[y_col])
    
    if x_outliers > 0 or y_outliers > 0:
        warnings.append(f"⚠️ Potential outliers detected: {x_col}: {x_outliers}, {y_col}: {y_outliers}")
        suggestions.append("Review outliers in the scatter plot - they may affect your results")
    
    # Check for constant values
    if clean_df[x_col].nunique() == 1:
        warnings.append(f"⚠️ {x_col} has constant values")
        suggestions.append("Choose a variable with more variation")
    
    if clean_df[y_col].nunique() == 1:
        warnings.append(f"⚠️ {y_col} has constant values")
        suggestions.append("Choose a variable with more variation")
    
    return warnings, suggestions

def create_correlation_heatmap(df):
    """Create correlation matrix heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        title="Correlation Matrix - All Numeric Variables",
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1]
    )
    
    fig.update_layout(
        height=400,
        title_x=0.5
    )
    
    return fig

def get_quality_indicator(value, thresholds, labels, colors):
    """Get traffic light indicator for quality metrics"""
    for i, threshold in enumerate(thresholds):
        if value >= threshold:
            return labels[i], colors[i]
    return labels[-1], colors[-1]

def display_pill_indicators_with_metrics(results):
    """Display pill indicators above each metric"""
    
    # Calculate indicators
    # Correlation strength
    abs_corr = abs(results['correlation'])
    corr_label, corr_color = get_quality_indicator(
        abs_corr,
        [0.8, 0.6, 0.4],
        ["🟢 Very Strong", "🟡 Strong", "🟠 Moderate", "🔴 Weak"],
        ["green", "orange", "orange", "red"]
    )
    
    # R-squared quality
    r2_label, r2_color = get_quality_indicator(
        results['r2'], 
        [0.7, 0.5, 0.3], 
        ["🟢 Excellent", "🟡 Good", "🟠 Fair", "🔴 Poor"],
        ["green", "orange", "orange", "red"]
    )
    
    # Statistical significance
    sig_label = "🟢 Significant" if results['p_value'] < 0.05 else "🔴 Not Significant"
    sig_color = "green" if results['p_value'] < 0.05 else "red"
    
    # Sample size adequacy
    n = results['n']
    size_label, size_color = get_quality_indicator(
        n,
        [100, 30, 10],
        ["🟢 Large", "🟡 Adequate", "🟠 Small", "🔴 Very Small"],
        ["green", "orange", "orange", "red"]
    )
    
    # Enhanced tooltips
    tooltips = enhanced_tooltips()
    
    # Display in columns with pill indicators above metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Pill indicator above correlation metric
        pill_class = f"pill-{corr_color}"
        # Extract just the quality level and emoji
        clean_label = corr_label.split(' ', 1)[1] if ' ' in corr_label else corr_label
        emoji = corr_label.split(' ')[0] if ' ' in corr_label else ""
        st.markdown(f'<div class="pill-indicator {pill_class}"><strong>Relationship:</strong> {clean_label} {emoji}</div>', unsafe_allow_html=True)
        st.metric("Correlation (r)", f"{results['correlation']:.3f}", help=tooltips['correlation'])
    
    with col2:
        # Pill indicator above R-squared metric
        pill_class = f"pill-{r2_color}"
        # Extract just the quality level and emoji
        clean_label = r2_label.split(' ', 1)[1] if ' ' in r2_label else r2_label
        emoji = r2_label.split(' ')[0] if ' ' in r2_label else ""
        st.markdown(f'<div class="pill-indicator {pill_class}"><strong>Model Quality:</strong> {clean_label} {emoji}</div>', unsafe_allow_html=True)
        st.metric("R-squared", f"{results['r2']:.3f}", help=tooltips['r_squared'])
    
    with col3:
        # Pill indicator above P-value metric
        pill_class = f"pill-{sig_color}"
        # Extract just the quality level and emoji
        clean_label = sig_label.split(' ', 1)[1] if ' ' in sig_label else sig_label
        emoji = sig_label.split(' ')[0] if ' ' in sig_label else ""
        st.markdown(f'<div class="pill-indicator {pill_class}"><strong>Significance:</strong> {clean_label} {emoji}</div>', unsafe_allow_html=True)
        st.metric("P-value", f"{results['p_value']:.4f}", help=tooltips['p_value'])
    
    with col4:
        # Pill indicator above Sample Size metric
        pill_class = f"pill-{size_color}"
        # Extract just the quality level and emoji
        clean_label = size_label.split(' ', 1)[1] if ' ' in size_label else size_label
        emoji = size_label.split(' ')[0] if ' ' in size_label else ""
        st.markdown(f'<div class="pill-indicator {pill_class}"><strong>Sample Size:</strong> {clean_label} {emoji}</div>', unsafe_allow_html=True)
        st.metric("Sample Size", f"{results['n']}", help=tooltips['sample_size'])

def create_prediction_calculator(results, x_col, y_col):
    """Create interactive prediction calculator with slider and persistent state"""
    st.markdown("###  Prediction Calculator")
    
    # Get reasonable range for input
    clean_df = results['clean_df']
    x_min, x_max = clean_df[x_col].min(), clean_df[x_col].max()
    x_mean = clean_df[x_col].mean()
    x_range = x_max - x_min
    
    # Extend the range slightly for more flexibility
    slider_min = x_min - (x_range * 0.2)
    slider_max = x_max + (x_range * 0.2)
    
    # Calculate a reasonable step size
    step_size = x_range / 100 if x_range > 0 else 0.01
    
    # Create unique identifier for this analysis
    analysis_id = f"{x_col}_vs_{y_col}"
    
    # Initialize the prediction value for this analysis if it doesn't exist
    if analysis_id not in st.session_state.prediction_values:
        st.session_state.prediction_values[analysis_id] = float(x_mean)
    
    # Create the slider in a styled container
    with st.container(height=200):
        
        # Display current range info inside the container
        st.markdown(f"**Adjust {x_col} value using the slider below:**")
        st.caption(f"Data range: {x_min:.2f} to {x_max:.2f} | Current range: {slider_min:.2f} to {slider_max:.2f}")
        
        # Create the slider
        x_input = st.slider(
            label=f"{x_col}",
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=st.session_state.prediction_values[analysis_id],
            step=float(step_size),
            help=f"Slide to select a value for {x_col}. The range extends beyond your data for extrapolation."
        )
    
    # Update session state if value changed
    if x_input != st.session_state.prediction_values[analysis_id]:
        st.session_state.prediction_values[analysis_id] = x_input
    
    # Make prediction and display results
    try:
        prediction = results['model'].predict([[x_input]])[0]
        
        # Calculate prediction interval (approximate)
        residuals = results['y'] - results['y_pred']
        residual_std = np.std(residuals)
        margin_of_error = 1.96 * residual_std  # Approximate 95% prediction interval
        
        # Display prediction results in columns
        st.markdown("**Prediction Results:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Value", f"{prediction:.2f}")
        
        with col2:
            st.metric("Lower Bound (95%)", f"{prediction - margin_of_error:.2f}")
        
        with col3:
            st.metric("Upper Bound (95%)", f"{prediction + margin_of_error:.2f}")
        
        # Interpretation
        # Determine if we're interpolating or extrapolating
        if x_min <= x_input <= x_max:
            prediction_type = "interpolation (within data range)"
            confidence_note = "This prediction is based on interpolation within your data range."
        else:
            prediction_type = "extrapolation (outside data range)"
            confidence_note = "⚠️ This prediction is based on extrapolation beyond your data range. Use with caution."
        
        st.info(f"💡 **Interpretation:** When {x_col} = {x_input:.2f}, the predicted {y_col} is **{prediction:.2f}** ")
        st.caption(f"Prediction method: {prediction_type}. {confidence_note}")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def enhanced_tooltips():
    """Add enhanced tooltips and help text"""
    return {
        'correlation': "Correlation measures the strength and direction of the linear relationship between two variables. Values range from -1 to +1.",
        'r_squared': "R-squared shows what percentage of the variation in the dependent variable is explained by the independent variable.",
        'p_value': "P-value indicates the probability that the observed relationship occurred by chance. Values < 0.05 are typically considered significant.",
        'sample_size': "The number of complete observations used in the analysis. Larger samples generally provide more reliable results."
    }

def perform_regression_analysis(df, x_col, y_col):
    """Perform comprehensive regression analysis"""
    # Remove rows with missing values
    clean_df = df[[x_col, y_col]].dropna()
    
    if len(clean_df) < 3:
        st.error("Not enough data points for regression analysis (minimum 3 required)")
        return None
    
    x = clean_df[x_col].values.reshape(-1, 1)
    y = clean_df[y_col].values
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(x, y)
    
    # Predictions
    y_pred = model.predict(x)
    
    # Calculate statistics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Pearson correlation
    correlation, p_value = stats.pearsonr(clean_df[x_col], clean_df[y_col])
    
    # Additional statistics
    n = len(clean_df)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Standard error of slope
    x_mean = np.mean(clean_df[x_col])
    ss_x = np.sum((clean_df[x_col] - x_mean) ** 2)
    se_slope = np.sqrt(mse / ss_x)
    
    # t-statistic for slope
    t_stat = slope / se_slope
    t_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    
    # Confidence intervals for slope
    t_critical = stats.t.ppf(0.975, n - 2)
    slope_ci_lower = slope - t_critical * se_slope
    slope_ci_upper = slope + t_critical * se_slope
    
    return {
        'model': model,
        'clean_df': clean_df,
        'x': x,
        'y': y,
        'y_pred': y_pred,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'correlation': correlation,
        'p_value': p_value,
        'n': n,
        'slope': slope,
        'intercept': intercept,
        'se_slope': se_slope,
        't_stat': t_stat,
        't_p_value': t_p_value,
        'slope_ci_lower': slope_ci_lower,
        'slope_ci_upper': slope_ci_upper
    }

def create_regression_plot(results, x_col, y_col):
    """Create an interactive regression plot"""
    clean_df = results['clean_df']
    
    # Create scatter plot with regression line
    fig = px.scatter(
        clean_df, 
        x=x_col, 
        y=y_col,
        title=f'Regression Analysis: {y_col} vs {x_col}',
        labels={x_col: x_col, y_col: y_col}
    )
    
    # Add regression line
    x_range = np.linspace(clean_df[x_col].min(), clean_df[x_col].max(), 100)
    y_range = results['slope'] * x_range + results['intercept']
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=2)
        )
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_residual_plots(results, x_col, y_col):
    """Create residual plots for model diagnostics"""
    clean_df = results['clean_df']
    residuals = results['y'] - results['y_pred']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Residuals vs Fitted', 'Q-Q Plot', 'Histogram of Residuals', 'Residuals vs X'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Residuals vs Fitted
    fig.add_trace(
        go.Scatter(x=results['y_pred'], y=residuals, mode='markers', name='Residuals vs Fitted'),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Q-Q Plot
    (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(residuals, dist="norm", plot=None)
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q Plot'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=osm, y=slope_qq * osm + intercept_qq, mode='lines', name='Q-Q Line'),
        row=1, col=2
    )
    
    # Histogram of Residuals
    fig.add_trace(
        go.Histogram(x=residuals, name='Residuals Distribution', nbinsx=20),
        row=2, col=1
    )
    
    # Residuals vs X
    fig.add_trace(
        go.Scatter(x=clean_df[x_col], y=residuals, mode='markers', name='Residuals vs X'),
        row=2, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def interpret_results(results, x_col, y_col):
    """Provide interpretation of regression results"""
    r2 = results['r2']
    correlation = results['correlation']
    p_value = results['p_value']
    slope = results['slope']
    t_p_value = results['t_p_value']
    
    interpretation = []
    
    # Strength of relationship
    if abs(correlation) >= 0.8:
        strength = "very strong"
    elif abs(correlation) >= 0.6:
        strength = "strong"
    elif abs(correlation) >= 0.4:
        strength = "moderate"
    elif abs(correlation) >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    
    # Direction of relationship
    direction = "positive" if correlation > 0 else "negative"
    interpretation.append(f"")
    interpretation.append(f"**Relationship Strength & Direction:** There is a {strength} {direction} relationship between {x_col} and {y_col} (r = {correlation:.3f}).")
    # R-squared interpretation
    interpretation.append(f"**Variance Explained:** {r2:.1%} of the variance in {y_col} is explained by {x_col}.")
    
    
    
    # Statistical significance
    alpha = 0.05
    if p_value < alpha:
        interpretation.append(f"**Statistical Significance:** The relationship is statistically significant (p = {p_value:.4f} < {alpha}), meaning it's unlikely to have occurred by chance.")
    else:
        interpretation.append(f"**Statistical Significance:** The relationship is not statistically significant (p = {p_value:.4f} ≥ {alpha}), meaning it could have occurred by chance.")
    
    # Slope interpretation
    if slope > 0:
        interpretation.append(f"**Practical Interpretation:** For every 1-unit increase in {x_col}, {y_col} increases by approximately {slope:.4f} units on average.")
    else:
        interpretation.append(f"**Practical Interpretation:** For every 1-unit increase in {x_col}, {y_col} decreases by approximately {abs(slope):.4f} units on average.")
    
    # Confidence in slope
    if t_p_value < 0.05:
        interpretation.append(f"**Slope Significance:** The slope is statistically significant (p = {t_p_value:.4f}), indicating a reliable relationship.")
    else:
        interpretation.append(f"**Slope Significance:** The slope is not statistically significant (p = {t_p_value:.4f}), indicating the relationship may not be reliable.")
    
    # Model quality assessment
    if r2 >= 0.7:
        quality = "excellent"
    elif r2 >= 0.5:
        quality = "good"
    elif r2 >= 0.3:
        quality = "fair"
    else:
        quality = "poor"
    
    interpretation.append(f"**Model Quality:** The model has {quality} predictive power based on the R² value of {r2:.3f}.")
    
    return interpretation

def main():
    st.title("Regression Analysis")
    
    st.info("""
    This tool helps you analyze relationships between variables in your data using linear regression.
    Upload your data file or try with sample data to explore relationships between variables.
    """)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown("## 📁 Data & Settings")
        
        # Load data
        df = load_data()
        
        if df is not None:
            # Get numeric columns
            numeric_cols = get_numeric_columns(df)
            
            if len(numeric_cols) < 2:
                st.error("Your dataset needs at least 2 numeric columns for regression analysis.")
                return
            
            # Variable selection
            st.markdown("---")
            st.markdown("### Variable Selection")
            
            x_col = st.selectbox(
                "Select Independent Variable (X)",
                numeric_cols,
                help="The variable you want to use to predict the other variable (the 'cause')"
            )
            
            y_col = st.selectbox(
                "Select Dependent Variable (Y)",
                [col for col in numeric_cols if col != x_col],
                help="The variable you want to predict or explain (the 'effect')"
            )
            
            st.markdown("<br>",unsafe_allow_html=True)
            analyze_button = st.button("🔍 Run Analysis", type="primary")
            
        else:
            
            # Enhanced example with more guidance
            st.markdown("<br>",  unsafe_allow_html=True)
            with st.expander("📋 What data works best?"):
                st.markdown("""
                **Your data should have:**
                - At least 2 numeric columns
                - At least 10 rows (preferably 30+)
                - No constant values
                - Minimal missing data
                
                **Good examples:**
                - TV Spend vs Website Traffic
                - AVG Home Value (from CensusLAB) vs Leads
                - Social Media Spend vs Lead Generation
                - Branded Impression Volume v Sales
                """)
                
                example_data = pd.DataFrame({
                    'TV_Spend': [50, 100, 150, 200, 250],
                    'Website_Traffic': [12, 24, 36, 48, 60],
                    'Sales_Revenue': [45, 68, 88, 105, 125]
                })
                st.dataframe(example_data)
            
            analyze_button = False
    
    # Main content area
    if df is not None:
        if x_col and y_col:
            # Create analysis identifier
            analysis_id = f"{x_col}_vs_{y_col}"
            
            # Check if we need to run analysis
            if analyze_button or (st.session_state.current_analysis == analysis_id and st.session_state.analysis_results is not None):
                
                # Only perform analysis if button clicked or if this is a new analysis
                if analyze_button or st.session_state.current_analysis != analysis_id:
                    # Perform analysis
                    st.session_state.analysis_results = perform_regression_analysis(df, x_col, y_col)
                    st.session_state.current_analysis = analysis_id
                
                results = st.session_state.analysis_results
                
                if results:
                    # Display main results
                    st.subheader("Analysis Results")
                    st.write("Metrics Summary:")
                    
                    # Display pill indicators above metrics instead of separate section
                    display_pill_indicators_with_metrics(results)
                    
                    # Apply metric card styling
                    style_metric_cards(
                        border_left_color="#6DC6DB", 
                        border_color="#FFFFFF", 
                        background_color=None
                    )
                    
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Create tabs for organized content
                    tab5, tab1, tab2, tab3, tab4 = st.tabs(["Insights", "Regression", "Prediction", "Data & Statistics", "Model Diagnostics"])
                    with tab5:
                        # Regression plot
                        st.markdown("### Analysis Insights")
                        # Interpretation with enhanced formatting
                        interpretation = interpret_results(results, x_col, y_col)
                            
                        interpretation_text = "\n\n".join(interpretation)
                        st.markdown(f'<div class="interpretation-box">{interpretation_text}</div>', unsafe_allow_html=True)
                        
                    with tab1:
                        # Regression plot
                        st.markdown("### Regression Plot")
                        fig = create_regression_plot(results, x_col, y_col)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Prediction calculator
                        create_prediction_calculator(results, x_col, y_col)
                    
                    with tab3:
                        # Data overview and summary statistics side by side
                        st.markdown("### Data Overview")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown("**Summary Statistics:**")
                            numeric_cols = get_numeric_columns(df)
                            st.dataframe(df[numeric_cols].describe().round(2))
                        
                        with col2:
                            # Correlation heatmap
                            fig_heatmap = create_correlation_heatmap(df)
                            if fig_heatmap:
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                                st.caption("💡 Look for strong correlations (dark red/blue) between variables")
                        
                        st.markdown("---")
                        
                        # Detailed regression statistics
                        st.markdown("### Detailed Regression Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Regression Equation:**")
                            st.latex(f"y = {results['slope']:.4f}x + {results['intercept']:.4f}")
                            
                            st.write("**Correlation Statistics:**")
                            st.write(f"- Pearson correlation: {results['correlation']:.4f}")
                            st.write(f"- P-value: {results['p_value']:.4f}")
                            st.write(f"- R-squared: {results['r2']:.4f}")
                            
                        with col2:
                            st.write("**Slope Statistics:**")
                            st.write(f"- Slope: {results['slope']:.4f}")
                            st.write(f"- Standard Error: {results['se_slope']:.4f}")
                            st.write(f"- t-statistic: {results['t_stat']:.4f}")
                            st.write(f"- P-value: {results['t_p_value']:.4f}")
                            st.write(f"- 95% CI: [{results['slope_ci_lower']:.4f}, {results['slope_ci_upper']:.4f}]")
                            
                            st.write("**Model Performance:**")
                            st.write(f"- RMSE: {results['rmse']:.4f}")
                            st.write(f"- MSE: {results['mse']:.4f}")
                    
                    with tab4:
                        # Residual analysis only
                        st.markdown("### Residual Plots")
                        fig_residuals = create_residual_plots(results, x_col, y_col)
                        st.plotly_chart(fig_residuals, use_container_width=True)
                        
                        st.markdown("""
                        **How to interpret residual plots:**
                        - **Residuals vs Fitted**: Should show random scatter around zero
                        - **Q-Q Plot**: Points should follow the diagonal line for normal residuals
                        - **Histogram**: Should be approximately bell-shaped
                        - **Residuals vs X**: Should show random scatter around zero
                        """)
                        
                else:
                    st.error("Analysis failed. Please check your data and try again.")
    
    elif df is None:
        # Enhanced placeholder content when no data is loaded
        st.markdown("### 🚀 Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Option 1: Try Sample Data
            1. **Check the box** "Try with sample data" in the sidebar
            2. **Choose a dataset** from the dropdown
            3. **Select variables** to analyze
            4. **Click analyze** to see results
            """)
        
        with col2:
            st.markdown("""
            #### Option 2: Upload Your Data
            1. **Upload your file** using the sidebar
            2. **Review data quality** warnings if any
            3. **Choose your variables** carefully
            4. **Run the analysis** and interpret results
            """)

if __name__ == "__main__":
    main()
