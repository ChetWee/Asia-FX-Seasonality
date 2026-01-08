import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

# -------------------------
# 1️⃣ Define helper functions
# -------------------------

def fetch_data(asset, start='2008-12-29'):
    """Fetch asset data from Yahoo Finance."""
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    data = yf.download(asset, start=start, end=today, interval='1d').xs(asset, level=1, axis=1)
    data['average'] = 4 / (data['Close'] + data['High'] + data['Low'] + data['Open'])  # Invert for USDXXX
    return data

def compute_weekly_returns(data):
    """Compute weekly returns."""
    returns = pd.DataFrame(np.log(data['average'] / data['average'].shift(1)))
    returns['year'] = returns.index.year
    returns['week'] = (returns.groupby('year').cumcount().floordiv(5) + 1).clip(upper=52)
    weekly = (
        returns
        .groupby(['year', 'week'])['average']
        .sum()
        .reset_index(name='weekly_return')
    )
    return weekly

def compute_weekly_medians(df, lookback=10):
    """Compute weekly medians for seasonal signal."""
    out = []
    for year in sorted(df['year'].unique()):
        past = df[df['year'].between(year - lookback, year - 1)]
        if past['year'].nunique() < lookback:
            continue
        medians = (
            past
            .groupby('week')['weekly_return']
            .median()
            .reset_index()
            .rename(columns={'weekly_return': 'weekly_median_10y'})
        )
        medians['year'] = year
        out.append(medians)
    return pd.concat(out, ignore_index=True)

def compute_ic(df):
    """Compute Information Coefficient (IC) per year."""
    ic_list = []
    for year, group in df.groupby('year'):
        if len(group) < 2:  # Need at least 2 weeks to compute correlation
            continue
        ic, pval = spearmanr(group['seasonal'], group['weekly_return'])
        ic_list.append({'year': year, 'IC': ic, 'pval': pval})
    return pd.DataFrame(ic_list)

# -------------------------
# 2️⃣ Streamlit App
# -------------------------

# Sidebar dropdown for asset selection
st.sidebar.title("Asia FX Seasonality")
asset = st.sidebar.selectbox(
    "Select Asset",
    options=["KRW=X", "CNY=X", "SGD=X", "THB=X", "MYR=X", "TWD=X"],
    index=0
)

# Fetch data and compute metrics
assets = fetch_data(asset)
weekly = compute_weekly_returns(assets)
weekly_medians = compute_weekly_medians(weekly, lookback=10)

# Add seasonal signal
weekly_medians['avg_weekly_median_10y'] = (
    weekly_medians.groupby('year')['weekly_median_10y'].transform('mean')
)
weekly_medians['seasonal'] = (
    weekly_medians['weekly_median_10y'] - weekly_medians['avg_weekly_median_10y']
)

# Merge seasonal signal into weekly returns
weekly_with_seasonal = weekly.merge(
    weekly_medians[['year', 'week', 'seasonal']],
    on=['year', 'week'],
    how='inner'
)

# Compute IC
ic_df = compute_ic(weekly_with_seasonal)

# -------------------------
# 3️⃣ Calculate Current Week
# -------------------------

current_year = pd.Timestamp.today().year
assets_current_year = assets[assets.index.year == current_year]
current_week = (len(assets_current_year) // 5) + 1  # Divide trading days by 5 and add 1 for the week number

# -------------------------
# 4️⃣ Prepare `weekly_data`
# -------------------------

weekly_data = weekly.copy()
weekly_data = weekly_data.merge(
    weekly_medians[['year', 'week', 'seasonal']],
    on=['year', 'week'],
    how='outer'  # Include all rows, even if missing seasonal signal
)

# -------------------------
# 5️⃣ Display Heatmaps
# -------------------------

st.title(f"Weekly Returns and Seasonal Signal for {asset}")

# Weekly Returns Heatmap
st.subheader("Weekly Returns Heatmap")

# Dynamically calculate the range for the last 5 years
last_5_years = list(range(current_year - 5, current_year))

# Filter data for the last 5 years
hist_years = weekly_data[weekly_data['year'].isin(last_5_years)]
avg_row = hist_years.groupby('week')['weekly_return'].mean().to_frame().T
avg_row.index = ['5-Year Average']
hist_rows = hist_years.pivot(index='year', columns='week', values='weekly_return').sort_index(ascending=False)

# Concatenate rows: 5-Year Average + historical weekly returns
heatmap_df_actual = pd.concat([avg_row, hist_rows]) * 100  # Convert to percentages

plt.figure(figsize=(16, 8))
cmap_actual = sns.diverging_palette(0, 130, s=100, l=40, as_cmap=True)
ax_actual = sns.heatmap(
    heatmap_df_actual,
    cmap=cmap_actual,
    center=0,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Actual Returns (%)'},
    annot=False
)

# Highlight the current week column
if current_week in heatmap_df_actual.columns:
    col_idx = list(heatmap_df_actual.columns).index(current_week)
    ax_actual.add_patch(Rectangle(
        (col_idx, 0),  # lower-left corner
        1,             # width = 1 column
        heatmap_df_actual.shape[0],  # height = all rows
        fill=False,
        edgecolor='black',
        linewidth=3
    ))

plt.title(f'Weekly Returns Heatmap (Red = Down, Green = Up) - Highlight Week {current_week}')
plt.xlabel('Week')
plt.ylabel('Year')
st.pyplot(plt)

# Seasonal Signal Heatmap
st.subheader("Seasonal Signal Heatmap")
seasonal_2026 = weekly_data[weekly_data['year'] == 2026][['week', 'seasonal']].set_index('week').T
seasonal_2026.index = ['2026 Seasonal']
heatmap_df_seasonal = pd.concat([seasonal_2026]) * 100  # Convert seasonal signal to percentages

plt.figure(figsize=(16, 2))
cmap_seasonal = sns.diverging_palette(0, 130, s=100, l=40, as_cmap=True)
ax_seasonal = sns.heatmap(
    heatmap_df_seasonal,
    cmap=cmap_seasonal,
    center=0,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={'label': 'Seasonal Signal (%)'},
    annot=False
)

# Highlight the current week column
if current_week in heatmap_df_seasonal.columns:
    col_idx = list(heatmap_df_seasonal.columns).index(current_week)
    ax_seasonal.add_patch(Rectangle(
        (col_idx, 0),  # lower-left corner
        1,             # width = 1 column
        heatmap_df_seasonal.shape[0],  # height = all rows
        fill=False,
        edgecolor='black',
        linewidth=3
    ))

plt.title(f'Seasonal Signal Heatmap (Red = Down, Green = Up) - Highlight Week {current_week}')
plt.xlabel('Week')
plt.ylabel('Seasonal')
st.pyplot(plt)

# -------------------------
# 6️⃣ Display IC Chart
# -------------------------

st.subheader("Information Coefficient (IC) Chart")
plt.figure(figsize=(10, 5))
plt.bar(ic_df['year'], ic_df['IC'], color='blue')
plt.axhline(0, color='black', linestyle='--')
plt.title('Information Coefficient (Spearman Rank) of Weekly Seasonals')
plt.xlabel('Year')
plt.ylabel('IC')
st.pyplot(plt)

# Display mean IC value
mean_ic = ic_df['IC'].mean()
st.write(f"**Average IC over all years:** {mean_ic:.3f}")