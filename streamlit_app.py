import pathlib
from datetime import datetime

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riskfolio as rp
import streamlit as st
from pypfopt import (
    DiscreteAllocation,
    EfficientFrontier,
    HRPOpt,
    expected_returns,
    plotting,
    risk_models,
)
from pypfopt.discrete_allocation import get_latest_prices
from vnstock import Listing, Quote


# --- Inlined helper functions ---


def inject_custom_success_styling():
    """Inject custom CSS styling for Streamlit success alerts with earth-tone theme."""
    st.html("""
<style>
div[data-testid="stAlert"][data-baseweb="notification"] {
    background-color: #D4D4D4 !important;
    border-color: #D4D4D4 !important;
    color: #56524D !important;
}
.stAlert {
    background-color: #D4D4D4 !important;
    border-color: #D4D4D4 !important;
    color: #56524D !important;
}
.stSuccess, .st-success {
    background-color: #D4D4D4 !important;
    border-color: #D4D4D4 !important;
    color: #56524D !important;
}
div[data-testid="stAlert"] > div {
    background-color: #D4D4D4 !important;
    color: #56524D !important;
}
div[data-testid="stAlert"] .stMarkdown {
    color: #56524D !important;
}
div[data-testid="stAlert"] p {
    color: #56524D !important;
}
.stMarkdownContainer {
    background-color: #76706C !important;
}
</style>
""")


@st.cache_data(max_entries=1)
def load_stock_symbols() -> list[str]:
    """Load all valid stock symbols from vnstock Listing API."""
    symbols_df = Listing().all_symbols()
    return sorted(symbols_df["symbol"].tolist())


@st.cache_data
def fetch_portfolio_stock_data(
    symbols: list[str],
    start_date_str: str,
    end_date_str: str,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Fetch historical stock data for multiple symbols via vnstock API."""
    all_data = {}

    for symbol in symbols:
        try:
            quote = Quote(symbol=symbol)
            historical_data = quote.history(
                start=start_date_str, end=end_date_str, interval=interval, to_df=True
            )

            if not historical_data.empty:
                if "time" not in historical_data.columns:
                    historical_data["time"] = historical_data.index

                all_data[symbol] = historical_data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")

    return all_data


def process_portfolio_price_data(
    all_historical_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Process historical data from multiple stocks into combined price dataframe."""
    combined_prices = pd.DataFrame()

    for symbol, data in all_historical_data.items():
        if not data.empty:
            if "time" not in data.columns:
                if hasattr(data.index, "name") and data.index.name is None:
                    data = data.reset_index()
                data = data.rename(columns={data.columns[0]: "time"})

            temp_df = data[["time", "close"]].copy()
            temp_df.rename(columns={"close": f"{symbol}_close"}, inplace=True)

            if combined_prices.empty:
                combined_prices = temp_df
            else:
                combined_prices = pd.merge(
                    combined_prices, temp_df, on="time", how="outer"
                )

    if combined_prices.empty:
        return combined_prices

    combined_prices = combined_prices.sort_values("time")
    combined_prices.set_index("time", inplace=True)

    close_price_columns = [col for col in combined_prices.columns if "_close" in col]
    prices_df = combined_prices[close_price_columns]
    prices_df.columns = [col.replace("_close", "") for col in close_price_columns]
    prices_df = prices_df.dropna()

    return prices_df


# --- Streamlit page configuration ---

st.set_page_config(
    page_title="Stock Portfolio Optimization", page_icon="", layout="wide"
)

inject_custom_success_styling()

# --- Sidebar: Symbol loading and selection ---

st.sidebar.header("Portfolio Configuration")

DEFAULT_SYMBOLS = ["REE", "FMC", "DHC", "VNM", "VCB", "BID", "HPG", "FPT"]

try:
    stock_symbols_list = load_stock_symbols()
except Exception:
    stock_symbols_list = DEFAULT_SYMBOLS

symbols = st.sidebar.multiselect(
    "Select ticker symbols:",
    options=stock_symbols_list,
    default=["REE", "FMC", "DHC"],
    placeholder="Choose stock symbols...",
    help="Select multiple stock symbols for portfolio optimization",
)

# --- Sidebar: Date range ---

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=pd.to_datetime("2024-01-01"),
        max_value=pd.to_datetime("today"),
    )

with col2:
    end_date = st.date_input(
        "End Date",
        value=pd.to_datetime("today") - pd.Timedelta(days=1),
        max_value=pd.to_datetime("today"),
    )

# --- Sidebar: Risk and visualization ---

risk_aversion = st.sidebar.number_input(
    "Risk Aversion Parameter", value=1.0, min_value=0.1, max_value=10.0, step=0.1
)

colormap_options = [
    "copper",
    "gist_heat",
    "Greys",
    "gist_yarg",
    "gist_gray",
    "cividis",
    "magma",
    "inferno",
    "plasma",
    "viridis",
]
colormap = st.sidebar.selectbox(
    "Scatter Plot Colormap",
    options=colormap_options,
    index=0,
    help="Choose the color scheme for the efficient frontier scatter plot",
)

interval = "1D"

start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# --- Main content ---

st.title("Stock Portfolio Optimization")
st.write("Optimize your portfolio using Modern Portfolio Theory")

# Validate inputs
if len(symbols) < 2:
    st.error("Please enter at least 2 ticker symbols.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# Fetch and process data
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("Fetching historical data...")

all_historical_data = fetch_portfolio_stock_data(
    symbols, start_date_str, end_date_str, interval
)

progress_bar.empty()
status_text.empty()

if not all_historical_data:
    st.error("No data was fetched for any symbol. Please check your inputs.")
    st.stop()

status_text.text("Processing data...")
prices_df = process_portfolio_price_data(all_historical_data)

if prices_df.empty:
    st.error("No valid price data after processing.")
    st.stop()

# --- Data Summary ---

st.header("Data Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Symbols", len(symbols))
with col2:
    st.metric("Data Points", len(prices_df))

with st.expander("View Price Data"):
    view_option = st.radio(
        "Display option:",
        ["First 5 rows", "Last 5 rows"],
        horizontal=True,
        key="price_data_view",
    )

    if view_option == "First 5 rows":
        st.dataframe(prices_df.head())
    else:
        st.dataframe(prices_df.tail())

    st.write(f"Shape: {prices_df.shape}")

# --- Portfolio Optimization ---

status_text.text("Calculating portfolio optimization...")
returns = prices_df.pct_change().dropna()
mu = expected_returns.mean_historical_return(prices_df)
S = risk_models.sample_cov(prices_df)

# Max Sharpe Ratio Portfolio
ef_tangent = EfficientFrontier(mu, S)
weights_tangent = ef_tangent.max_sharpe()
weights_max_sharpe = ef_tangent.clean_weights()
ret_tangent, std_tangent, sharpe = ef_tangent.portfolio_performance()

# Min Volatility Portfolio
ef_min_vol = EfficientFrontier(mu, S)
ef_min_vol.min_volatility()
weights_min_vol = ef_min_vol.clean_weights()
ret_min_vol, std_min_vol, sharpe_min_vol = ef_min_vol.portfolio_performance()

# Max Utility Portfolio
ef_max_utility = EfficientFrontier(mu, S)
ef_max_utility.max_quadratic_utility(risk_aversion=risk_aversion, market_neutral=False)
weights_max_utility = ef_max_utility.clean_weights()
ret_utility, std_utility, sharpe_utility = ef_max_utility.portfolio_performance()

status_text.empty()

# --- Performance Metrics ---

st.header("Portfolio Optimization Results")

st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Max Sharpe Portfolio", f"{sharpe:.4f}", f"Return: {(ret_tangent * 100):.1f}%"
    )

with col2:
    st.metric(
        "Min Volatility Portfolio",
        f"{sharpe_min_vol:.4f}",
        f"Return: {(ret_min_vol * 100):.1f}%",
    )

with col3:
    st.metric(
        "Max Utility Portfolio",
        f"{sharpe_utility:.4f}",
        f"Return: {(ret_utility * 100):.1f}%",
    )

# --- Strategy selection (shared across tabs) ---

portfolio_choice = st.radio(
    "Select Portfolio Strategy:",
    ["Max Sharpe Portfolio", "Min Volatility Portfolio", "Max Utility Portfolio"],
    help="This selection applies to Dollar Allocation, Report, and Risk Analysis tabs",
    horizontal=True,
    key="portfolio_strategy_master",
)

# --- Tabs ---

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Efficient Frontier & Weights",
    "Hierarchical Risk Parity",
    "Dollars Allocation",
    "Report",
    "Risk Analysis",
])

with tab1:
    st.subheader("Efficient Frontier Analysis")
    fig, ax = plt.subplots(figsize=(12, 8))

    ef_plot = EfficientFrontier(mu, S)
    plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)

    ax.scatter(
        std_tangent,
        ret_tangent,
        marker="*",
        s=200,
        c="red",
        label="Max Sharpe",
        zorder=5,
    )
    ax.scatter(
        std_min_vol,
        ret_min_vol,
        marker="*",
        s=200,
        c="green",
        label="Min Volatility",
        zorder=5,
    )
    ax.scatter(
        std_utility,
        ret_utility,
        marker="*",
        s=200,
        c="blue",
        label="Max Utility",
        zorder=5,
    )

    n_samples = 5000
    w = np.random.dirichlet(np.ones(ef_plot.n_assets), n_samples)
    rets = w.dot(ef_plot.expected_returns)
    stds = np.sqrt(np.diag(w @ ef_plot.cov_matrix @ w.T))
    sharpes = rets / stds

    scatter = ax.scatter(stds, rets, marker=".", c=sharpes, cmap=colormap, alpha=0.6)
    plt.colorbar(scatter, label="Sharpe Ratio")

    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.set_xlabel("Annual Volatility")
    ax.set_ylabel("Annual Return")
    ax.legend()

    st.pyplot(fig)

    # Portfolio Weights
    st.subheader("Portfolio Weights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Max Sharpe Portfolio**")
        weights_df = pd.DataFrame(
            list(weights_max_sharpe.items()), columns=["Symbol", "Weight"]
        )
        weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_df, hide_index=True)

    with col2:
        st.write("**Min Volatility Portfolio**")
        weights_df = pd.DataFrame(
            list(weights_min_vol.items()), columns=["Symbol", "Weight"]
        )
        weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_df, hide_index=True)

    with col3:
        st.write("**Max Utility Portfolio**")
        weights_df = pd.DataFrame(
            list(weights_max_utility.items()), columns=["Symbol", "Weight"]
        )
        weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_df, hide_index=True)

    # Weight visualization
    st.subheader("Portfolio Weights Visualization")

    pie_colors = ["#56524D", "#76706C", "#AAA39F"]

    def create_pie_chart(weights_dict, title, colors):
        """Create an Altair pie chart for portfolio weights."""
        data = pd.DataFrame(list(weights_dict.items()), columns=["Symbol", "Weight"])
        data = data[data["Weight"] > 0.01]
        data = data.sort_values("Weight", ascending=False)

        if len(data) == 0:
            return None

        data["color"] = (
            colors[: len(data)]
            if len(data) <= len(colors)
            else colors + ["#D3D3D3"] * (len(data) - len(colors))
        )

        chart = (
            alt.Chart(data)
            .mark_arc(innerRadius=50, stroke="white", strokeWidth=2)
            .encode(
                theta=alt.Theta("Weight:Q", title="Weight"),
                color=alt.Color(
                    "Symbol:N",
                    scale=alt.Scale(range=data["color"].tolist()),
                    legend=alt.Legend(title="Symbols"),
                ),
                tooltip=[
                    alt.Tooltip("Symbol:N", title="Symbol"),
                    alt.Tooltip("Weight:Q", title="Weight", format=".2%"),
                ],
            )
            .properties(width=350, height=350, title=title)
        )

        return chart

    col1, col2, col3 = st.columns(3)

    with col1:
        pie1 = create_pie_chart(weights_max_sharpe, "Max Sharpe Portfolio", pie_colors)
        if pie1:
            st.altair_chart(pie1, width="stretch")
        else:
            st.write("No significant weights in Max Sharpe Portfolio")

    with col2:
        pie2 = create_pie_chart(weights_min_vol, "Min Volatility Portfolio", pie_colors)
        if pie2:
            st.altair_chart(pie2, width="stretch")
        else:
            st.write("No significant weights in Min Volatility Portfolio")

    with col3:
        pie3 = create_pie_chart(
            weights_max_utility, "Max Utility Portfolio", pie_colors
        )
        if pie3:
            st.altair_chart(pie3, width="stretch")
        else:
            st.write("No significant weights in Max Utility Portfolio")

    # Detailed performance table
    st.subheader("Detailed Performance Analysis")
    performance_df = pd.DataFrame({
        "Portfolio": ["Max Sharpe", "Min Volatility", "Max Utility"],
        "Expected Return": [
            f"{ret_tangent:.4f}",
            f"{ret_min_vol:.4f}",
            f"{ret_utility:.4f}",
        ],
        "Volatility": [
            f"{std_tangent:.4f}",
            f"{std_min_vol:.4f}",
            f"{std_utility:.4f}",
        ],
        "Sharpe Ratio": [
            f"{sharpe:.4f}",
            f"{sharpe_min_vol:.4f}",
            f"{sharpe_utility:.4f}",
        ],
    })
    st.dataframe(performance_df, hide_index=True)

with tab2:
    hrp = HRPOpt(returns=returns)
    weights_hrp = hrp.optimize()

    st.subheader("HRP Portfolio Weights")
    weights_df = pd.DataFrame(list(weights_hrp.items()), columns=["Symbol", "Weight"])
    weights_df["Weight"] = weights_df["Weight"].apply(lambda x: f"{x:.2%}")
    st.dataframe(weights_df, hide_index=True)

    st.subheader("HRP Dendrogram")
    fig_dendro, ax_dendro = plt.subplots(figsize=(12, 8))
    plotting.plot_dendrogram(hrp, ax=ax_dendro, show_tickers=True)
    st.pyplot(fig_dendro)

with tab3:
    st.subheader("Discrete Portfolio Allocation")

    portfolio_value = st.number_input(
        "Portfolio Value (VND)",
        min_value=1000000,
        max_value=100000000000,
        value=100000000,
        step=1000000,
        help="Enter your total portfolio value in Vietnamese Dong (VND)",
    )

    symbol_display = ", ".join(symbols[:3]) + ("..." if len(symbols) > 3 else "")
    st.info(f"**Using Strategy**: {portfolio_choice} | **Symbols**: {symbol_display}")

    # Get the selected weights
    if portfolio_choice == "Max Sharpe Portfolio":
        selected_weights = weights_max_sharpe
        portfolio_label = "Max Sharpe"
    elif portfolio_choice == "Min Volatility Portfolio":
        selected_weights = weights_min_vol
        portfolio_label = "Min Volatility"
    else:
        selected_weights = weights_max_utility
        portfolio_label = "Max Utility"

    if st.button("Calculate Allocation", key="discrete_allocation"):
        try:
            latest_prices = get_latest_prices(prices_df)
            latest_prices_actual = latest_prices * 1000

            da = DiscreteAllocation(
                selected_weights,
                latest_prices_actual,
                total_portfolio_value=portfolio_value,
            )
            allocation, leftover = da.greedy_portfolio()

            st.success(
                f"Allocation calculated successfully for {portfolio_label} Portfolio!"
            )

            st.subheader("Stock Allocation")
            allocation_df = pd.DataFrame(
                list(allocation.items()), columns=["Symbol", "Shares"]
            )
            allocation_df["Latest Price (VND)"] = allocation_df["Symbol"].map(
                latest_prices_actual
            )
            allocation_df["Total Value (VND)"] = (
                allocation_df["Shares"] * allocation_df["Latest Price (VND)"]
            )
            allocation_df["Weight %"] = (
                allocation_df["Total Value (VND)"] / portfolio_value * 100
            ).round(2)

            allocation_df["Latest Price (VND)"] = allocation_df[
                "Latest Price (VND)"
            ].apply(lambda x: f"{x:,.0f}")
            allocation_df["Total Value (VND)"] = allocation_df[
                "Total Value (VND)"
            ].apply(lambda x: f"{x:,.0f}")

            st.dataframe(allocation_df, hide_index=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                allocated_value = portfolio_value - leftover
                st.metric(
                    "Allocated Amount",
                    f"{allocated_value:,.0f} VND",
                    f"{(allocated_value / portfolio_value * 100):.1f}% of portfolio",
                )

            with col2:
                st.metric(
                    "Leftover Cash",
                    f"{leftover:,.0f} VND",
                    f"{(leftover / portfolio_value * 100):.1f}% of portfolio",
                )

            with col3:
                total_stocks = len(allocation)
                st.metric("Stocks to Buy", total_stocks)

            st.subheader("Investment Summary")
            st.info(f"""
            **Portfolio Strategy**: {portfolio_label}
            **Total Investment**: {portfolio_value:,.0f} VND
            **Allocated**: {allocated_value:,.0f} VND ({(allocated_value / portfolio_value * 100):.1f}%)
            **Remaining Cash**: {leftover:,.0f} VND ({(leftover / portfolio_value * 100):.1f}%)
            **Number of Stocks**: {total_stocks} stocks
            """)

        except Exception as e:
            st.error(f"Error calculating allocation: {str(e)}")
            st.error(
                "Please ensure you have selected stocks and loaded price data first."
            )
    else:
        st.info(
            "Click 'Calculate Allocation' to see how many shares to buy for each stock based on your selected portfolio strategy and investment amount."
        )

with tab4:
    st.subheader("Portfolio Excel Report Generator")
    st.write(
        "Generate comprehensive Excel reports for your optimized portfolios using Riskfolio-lib."
    )

    st.info(f"**Current Strategy**: {portfolio_choice}")

    if st.button("Generate Report", key="generate_excel_report"):
        project_root = pathlib.Path(
            pathlib.Path(pathlib.Path(__file__).resolve()).parent
        ).parent
        reports_dir = project_root / "exports" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        if portfolio_choice == "Max Sharpe Portfolio":
            selected_weights = weights_max_sharpe
            portfolio_name = "Max_Sharpe_Portfolio"
            portfolio_label = "Max Sharpe"
        elif portfolio_choice == "Min Volatility Portfolio":
            selected_weights = weights_min_vol
            portfolio_name = "Min_Volatility_Portfolio"
            portfolio_label = "Min Volatility"
        else:
            selected_weights = weights_max_utility
            portfolio_name = "Max_Utility_Portfolio"
            portfolio_label = "Max Utility"

        selected_weights_df = pd.DataFrame.from_dict(
            selected_weights, orient="index", columns=[portfolio_name]
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{portfolio_name}_{timestamp}"
        filepath_base = reports_dir / filename_base

        rp.excel_report(returns=returns, w=selected_weights_df, name=filepath_base)

        st.success("Excel report generated successfully!")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Portfolio**: {portfolio_label}")
        with col2:
            st.info(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        filepath_xlsx = pathlib.Path(str(filepath_base) + ".xlsx")
        with filepath_xlsx.open("rb") as file:
            st.download_button(
                label="Download Excel Report",
                data=file.read(),
                file_name=filename_base + ".xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=f"Download the {portfolio_label} portfolio Excel report",
            )

        file_size = pathlib.Path(filepath_xlsx).stat().st_size / 1024
        st.caption(f"File: {filename_base}.xlsx ({file_size:.1f} KB)")

    else:
        st.info(
            "Select a portfolio strategy and click 'Generate Report' to create a comprehensive Excel analysis."
        )

        st.markdown("### Report Contents")
        st.markdown("""
        The Excel report will include:
        - **Portfolio Weights**: Detailed allocation percentages
        - **Performance Metrics**: Returns, volatility, and Sharpe ratio
        - **Risk Analysis**: Comprehensive risk assessment
        - **Asset Statistics**: Individual asset performance data
        - **Correlation Matrix**: Asset correlation analysis
        """)

with tab5:
    st.subheader("Risk Analysis Table")

    if portfolio_choice == "Max Sharpe Portfolio":
        selected_weights = weights_max_sharpe
        portfolio_label = "Max Sharpe"
    elif portfolio_choice == "Min Volatility Portfolio":
        selected_weights = weights_min_vol
        portfolio_label = "Min Volatility"
    else:
        selected_weights = weights_max_utility
        portfolio_label = "Max Utility"

    symbol_display = ", ".join(symbols[:3]) + ("..." if len(symbols) > 3 else "")
    st.info(
        f"**Analyzing Strategy**: {portfolio_choice} | **Symbols**: {symbol_display}"
    )

    weights_df = pd.DataFrame.from_dict(
        selected_weights, orient="index", columns=["Weights"]
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax = rp.plot_table(
        returns=returns,
        w=weights_df,
        MAR=0,
        alpha=0.05,
        ax=ax,
    )

    st.pyplot(fig)

    # Drawdown Analysis
    st.subheader("Portfolio Drawdown Analysis")

    fig_drawdown, ax_drawdown = plt.subplots(figsize=(12, 8))

    ax_drawdown = rp.plot_drawdown(
        returns=returns,
        w=weights_df,
        alpha=0.05,
        kappa=0.3,
        solver="CLARABEL",
        height=8,
        width=10,
        height_ratios=[2, 3],
        ax=ax_drawdown,
    )

    st.pyplot(fig_drawdown)

    # Portfolio Returns Risk Measures
    st.subheader("Portfolio Returns Risk Measures")

    fig_range, ax_range = plt.subplots(figsize=(12, 6))

    ax_range = rp.plot_range(
        returns=returns,
        w=weights_df,
        alpha=0.05,
        a_sim=100,
        beta=None,
        b_sim=None,
        bins=50,
        height=6,
        width=10,
        ax=ax_range,
    )

    st.pyplot(fig_range)

    with st.expander("Understanding the Risk Analysis Table"):
        st.markdown(f"""
        This table provides comprehensive risk metrics for your {portfolio_label} portfolio:

        **Key Metrics:**
        - **Expected Return**: Annualized expected portfolio return
        - **Volatility**: Portfolio standard deviation (risk measure)
        - **Sharpe Ratio**: Risk-adjusted return measure
        - **VaR**: Value at Risk - potential loss at 95% confidence
        - **CVaR**: Conditional Value at Risk - expected loss beyond VaR
        - **Max Drawdown**: Largest peak-to-trough decline
        - **Calmar Ratio**: Return to max drawdown ratio

        *Generated using riskfolio-lib risk analysis framework*
        """)

# Footer
st.markdown("---")
st.markdown(
    "*Portfolio optimization based on Modern Portfolio Theory. Past performance does not guarantee future results.*"
)
