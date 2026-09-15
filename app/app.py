import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL LIGHT UI
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APPLICATION
       ======================================================== */

    .stApp {
        background-color: #FFFFFF;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #F4F8FC;
        border-right: 1px solid #D9E2EC;
    }

    [data-testid="stSidebar"] h1 {
        color: #123B66 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #123B66 !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] p {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] label {
        color: #000000 !important;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        color: #123B66 !important;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        color: #000000 !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */
    [data-testid="stHeading"] {
        color: #000000 !important;
    }
    .section-heading {
        color: #123B66 !important;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #C9D8E8 !important;
        border-left: 4px solid #2F75B5 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }

    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }


    /* ========================================================
       SIDEBAR SELECTBOX
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       SIDEBAR SLIDER
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stSlider"] label {
        color: #000000 !important;
        font-weight: 600 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    [data-testid="stSidebar"] button {
        background-color: #2F75B5 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] button p {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: #1F5A8A !important;
        color: #FFFFFF !important;
    }


    /* ========================================================
       DATAFRAME CONTAINER
       ======================================================== */

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #B8CCE0 !important;
        border-radius: 6px !important;
    }


    /* ========================================================
       EXPANDER
       ======================================================== */

    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #B8CCE0 !important;
        border-radius: 8px !important;
    }

    [data-testid="stExpander"] summary {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary span {
        color: #000000 !important;
    }

    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }

    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }

    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] strong {
        color: #000000 !important;
    }


    /* ========================================================
       ALERT BOXES
       ======================================================== */

    [data-testid="stAlert"] {
        color: #000000 !important;
    }

    [data-testid="stAlert"] p {
        color: #000000 !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #D9E2EC !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        color: #000000 !important;
        text-align: center;
        font-size: 0.8rem;
        padding-top: 25px;
        margin-top: 30px;
        border-top: 1px solid #D9E2EC;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "retail_cleaned.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

FEATURE_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ============================================================
# AVAILABLE PRODUCTS
# ============================================================

PRODUCTS = {

    "21212":
        "PACK OF 72 RETRO SPOT CAKE CASES",

    "85123A":
        "WHITE HANGING HEART T-LIGHT HOLDER",

    "84077":
        "WORLD WAR 2 GLIDERS ASSTD DESIGNS",

    "85099B":
        "JUMBO BAG RED WHITE SPOTTY",

    "17003":
        "BROCADE RING PURSE"
}


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [

    "DayOfWeek",

    "DayOfMonth",

    "Month",

    "Quarter",

    "WeekOfYear",

    "IsWeekend",

    "Lag_1",

    "Lag_7",

    "Lag_14",

    "Lag_28",

    "Rolling_Mean_7",

    "Rolling_Mean_14",

    "Rolling_Mean_28"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(product_code):

    model_path = (
        MODEL_DIR
        / f"random_forest_product_{product_code}.joblib"
    )

    return joblib.load(model_path)


# ============================================================
# LOAD PRODUCT FEATURE DATA
# ============================================================

@st.cache_data
def load_product_data(product_code):

    file_path = (
        FEATURE_DIR
        / f"product_{product_code}_features.csv"
    )

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD CLEANED DATA
# ============================================================

@st.cache_data
def load_cleaned_data():

    df = pd.read_csv(
        DATA_PATH
    )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"]
    )

    return df


# ============================================================
# RECURSIVE MULTI-DAY FORECASTING
# ============================================================

def forecast_next_days(
    model,
    product_data,
    number_of_days
):

    history = product_data[
        [
            "Date",
            "Demand"
        ]
    ].copy()

    history = (
        history
        .sort_values("Date")
        .reset_index(drop=True)
    )

    forecasts = []


    # --------------------------------------------------------
    # Generate each future day
    # --------------------------------------------------------

    for _ in range(number_of_days):

        next_date = (
            history["Date"].iloc[-1]
            + pd.Timedelta(days=1)
        )

        demand_values = (
            history["Demand"].tolist()
        )


        # ----------------------------------------------------
        # Lag features
        # ----------------------------------------------------

        lag_1 = (
            demand_values[-1]
        )

        lag_7 = (
            demand_values[-7]
            if len(demand_values) >= 7
            else 0
        )

        lag_14 = (
            demand_values[-14]
            if len(demand_values) >= 14
            else 0
        )

        lag_28 = (
            demand_values[-28]
            if len(demand_values) >= 28
            else 0
        )


        # ----------------------------------------------------
        # Rolling mean features
        # ----------------------------------------------------

        rolling_mean_7 = np.mean(
            demand_values[-7:]
        )

        rolling_mean_14 = np.mean(
            demand_values[-14:]
        )

        rolling_mean_28 = np.mean(
            demand_values[-28:]
        )


        # ----------------------------------------------------
        # Create model input
        # ----------------------------------------------------

        features = pd.DataFrame(
            [{
                "DayOfWeek":
                    next_date.dayofweek,

                "DayOfMonth":
                    next_date.day,

                "Month":
                    next_date.month,

                "Quarter":
                    next_date.quarter,

                "WeekOfYear":
                    int(
                        next_date.isocalendar().week
                    ),

                "IsWeekend":
                    int(
                        next_date.dayofweek >= 5
                    ),

                "Lag_1":
                    lag_1,

                "Lag_7":
                    lag_7,

                "Lag_14":
                    lag_14,

                "Lag_28":
                    lag_28,

                "Rolling_Mean_7":
                    rolling_mean_7,

                "Rolling_Mean_14":
                    rolling_mean_14,

                "Rolling_Mean_28":
                    rolling_mean_28
            }]
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            features[FEATURES]
        )[0]


        # Demand cannot be negative

        prediction = max(
            0,
            prediction
        )


        # ----------------------------------------------------
        # Store prediction
        # ----------------------------------------------------

        forecasts.append(
            {
                "Date":
                    next_date,

                "Predicted Demand":
                    prediction
            }
        )


        # ----------------------------------------------------
        # Recursive update
        # ----------------------------------------------------

        history = pd.concat(
            [
                history,

                pd.DataFrame(
                    {
                        "Date":
                            [next_date],

                        "Demand":
                            [prediction]
                    }
                )
            ],
            ignore_index=True
        )


    return pd.DataFrame(
        forecasts
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "Retail Analytics"
    )

    st.caption(
        "Demand Forecasting System"
    )

    st.divider()


    # --------------------------------------------------------
    # Forecast Settings
    # --------------------------------------------------------

    st.subheader(
        "Forecast Settings"
    )

    product_code = st.selectbox(

        "Select Product",

        options=list(
            PRODUCTS.keys()
        ),

        format_func=lambda code:
            f"{code} — {PRODUCTS[code]}"
    )


    forecast_days = st.slider(

        "Forecast Horizon",

        min_value=1,

        max_value=30,

        value=7,

        step=1
    )


    generate_forecast = st.button(

        "Generate Forecast",

        use_container_width=True,

        type="primary"
    )


    st.divider()


    # --------------------------------------------------------
    # Model Information
    # --------------------------------------------------------

    st.subheader(
        "Model"
    )

    st.write(
        "Random Forest Regressor"
    )

    st.caption(
        "300 trees"
    )

    st.caption(
        "Maximum depth = 10"
    )

    st.caption(
        "Minimum samples per leaf = 2"
    )


    st.divider()


    # --------------------------------------------------------
    # Forecasting Method
    # --------------------------------------------------------

    st.subheader(
        "Forecasting"
    )

    st.caption(
        "Recursive multi-day prediction"
    )


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Retail Inventory Demand Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Seasonal analysis and machine learning based demand forecasting'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:

    model = load_model(
        product_code
    )

    product_data = load_product_data(
        product_code
    )

    cleaned_data = load_cleaned_data()

except Exception as e:

    st.error(
        f"Unable to load project data or model: {e}"
    )

    st.stop()


# ============================================================
# FIND PRODUCT HISTORY
# ============================================================

product_rows = cleaned_data[
    cleaned_data["StockCode"].astype(str)
    == str(product_code)
]


if len(product_rows) > 0:

    first_sale = (
        product_rows["InvoiceDate"]
        .min()
    )

    last_sale = (
        product_rows["InvoiceDate"]
        .max()
    )

else:

    first_sale = (
        product_data["Date"]
        .min()
    )

    last_sale = (
        product_data["Date"]
        .max()
    )


# ============================================================
# SELECTED PRODUCT
# ============================================================

st.markdown(
    f'<h3 style="color: black;">{product_code} — {PRODUCTS[product_code]}</h3>',
    unsafe_allow_html=True
)

st.caption(
    "Selected Product"
)


# ============================================================
# FORECAST
# ============================================================

forecast_df = forecast_next_days(

    model,

    product_data,

    forecast_days
)


# ============================================================
# FORECAST KPIs
# ============================================================

total_forecast = (
    forecast_df[
        "Predicted Demand"
    ].sum()
)


average_forecast = (
    forecast_df[
        "Predicted Demand"
    ].mean()
)


peak_index = (
    forecast_df[
        "Predicted Demand"
    ].idxmax()
)


peak_demand = (
    forecast_df.loc[
        peak_index,
        "Predicted Demand"
    ]
)


peak_date = (
    forecast_df.loc[
        peak_index,
        "Date"
    ]
)


historical_total = (
    product_data[
        "Demand"
    ].sum()
)


model_ready_days = (
    len(product_data)
)


# ============================================================
# FORECAST SUMMARY
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Forecast Summary'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Forecast Horizon",
        f"{forecast_days} Days"
    )


with col2:

    st.metric(
        "Total Forecast",
        f"{total_forecast:,.0f} units"
    )


with col3:

    st.metric(
        "Average Daily Demand",
        f"{average_forecast:,.0f} units"
    )


with col4:

    st.metric(
        "Peak Forecast",
        f"{peak_demand:,.0f} units"
    )


# ============================================================
# HISTORICAL OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Historical Overview'
    '</div>',
    unsafe_allow_html=True
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "First Recorded Sale",
        first_sale.strftime(
            "%d %b %Y"
        )
    )


with col2:

    st.metric(
        "Latest Recorded Sale",
        last_sale.strftime(
            "%d %b %Y"
        )
    )


with col3:

    st.metric(
        "Model-Ready Days",
        f"{model_ready_days:,}"
    )


with col4:

    st.metric(
        "Historical Demand",
        f"{historical_total:,.0f} units"
    )


# ============================================================
# DEMAND TREND & FORECAST
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Demand Trend & Forecast'
    '</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Use the latest 90 historical days
# ------------------------------------------------------------

historical_plot = (
    product_data
    .tail(90)
    .copy()
)


# ------------------------------------------------------------
# Create chart
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(14, 5),
    facecolor="white"
)

ax.set_facecolor(
    "white"
)


# ------------------------------------------------------------
# Historical demand
# ------------------------------------------------------------

ax.plot(
    historical_plot["Date"],
    historical_plot["Demand"],
    linewidth=2,
    label="Historical Demand"
)


# ------------------------------------------------------------
# Forecast demand
# ------------------------------------------------------------

ax.plot(
    forecast_df["Date"],
    forecast_df["Predicted Demand"],
    linewidth=2.5,
    linestyle="--",
    label="Forecast"
)


# ------------------------------------------------------------
# Forecast start
# ------------------------------------------------------------

forecast_start = (
    forecast_df[
        "Date"
    ].iloc[0]
)


ax.axvline(
    forecast_start,
    linestyle=":",
    linewidth=1.5,
    label="Forecast Start"
)


# ------------------------------------------------------------
# Axis labels
# ------------------------------------------------------------

ax.set_xlabel(
    "Date",
    color="black"
)

ax.set_ylabel(
    "Demand (Units)",
    color="black"
)


# ------------------------------------------------------------
# Chart title
# ------------------------------------------------------------

ax.set_title(
    "Historical Demand and Future Forecast",
    color="black",
    fontsize=14,
    fontweight="bold"
)


# ------------------------------------------------------------
# Tick colors
# ------------------------------------------------------------

ax.tick_params(
    axis="both",
    colors="black"
)


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.grid(
    alpha=0.20,
    linestyle="--"
)


# ------------------------------------------------------------
# Legend
# ------------------------------------------------------------

legend = ax.legend()

for text in legend.get_texts():

    text.set_color(
        "black"
    )


# ------------------------------------------------------------
# Date formatting
# ------------------------------------------------------------

fig.autofmt_xdate()


plt.tight_layout()


st.pyplot(
    fig,
    use_container_width=True
)


plt.close(fig)


# ============================================================
# FORECAST SCHEDULE
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Forecast Schedule'
    '</div>',
    unsafe_allow_html=True
)


display_forecast = (
    forecast_df.copy()
)


# ------------------------------------------------------------
# Forecast day number
# ------------------------------------------------------------

display_forecast.insert(
    0,
    "Day",
    range(
        1,
        len(display_forecast) + 1
    )
)


# ------------------------------------------------------------
# Format date
# ------------------------------------------------------------

display_forecast["Date"] = (
    display_forecast["Date"]
    .dt.strftime(
        "%d %b %Y"
    )
)


# ------------------------------------------------------------
# Round predicted demand
# ------------------------------------------------------------

display_forecast[
    "Predicted Demand"
] = (
    display_forecast[
        "Predicted Demand"
    ]
    .round(0)
    .astype(int)
)


# ------------------------------------------------------------
# Rename column
# ------------------------------------------------------------

display_forecast = (
    display_forecast.rename(
        columns={
            "Predicted Demand":
                "Predicted Demand (Units)"
        }
    )
)


# ------------------------------------------------------------
# Display forecast table
# ------------------------------------------------------------

st.dataframe(
    display_forecast,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BUSINESS PLANNING INSIGHT
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Business Planning Insight'
    '</div>',
    unsafe_allow_html=True
)


if peak_demand > average_forecast * 1.25:

    insight = (
        f"Peak demand is expected to reach "
        f"{peak_demand:,.0f} units on "
        f"{peak_date.strftime('%d %b %Y')}. "
        f"Consider additional inventory coverage "
        f"around this period."
    )


elif peak_demand < average_forecast * 1.10:

    insight = (
        "Demand is expected to remain relatively "
        "stable throughout the selected forecast "
        "horizon."
    )


else:

    insight = (
        f"Demand is expected to average "
        f"{average_forecast:,.0f} units per day "
        f"with moderate variation during the "
        f"forecast period."
    )


st.info(
    insight
)


st.warning(
    "Forecasts are model estimates. Inventory "
    "decisions should also consider current stock, "
    "supplier lead time, promotions, and other "
    "business factors."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Model Information'
    '</div>',
    unsafe_allow_html=True
)


with st.expander(
    "View model configuration and features"
):

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            "**Algorithm**"
        )

        st.write(
            "Random Forest Regressor"
        )


    with col2:

        st.markdown(
            "**Configuration**"
        )

        st.write(
            "300 trees • Depth 10 • Leaf 2"
        )


    with col3:

        st.markdown(
            "**Forecasting Method**"
        )

        st.write(
            "Recursive multi-day forecasting"
        )


    st.divider()


    st.markdown(
        "**Features used by the model:**"
    )


    st.write(
        ", ".join(FEATURES)
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Retail Inventory Demand Prediction System
        • Machine Learning
        • Seasonal Analysis
        • Demand Forecasting
    </div>
    """,
    unsafe_allow_html=True
)