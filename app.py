import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Trading Journal", layout="wide")

# --- Load and clean CSV ---
@st.cache_data
def load_trades(file_path):
    # Fix header if needed
    with open(file_path, 'r') as f:
        lines = f.readlines()
    if not lines[0].startswith("symbol"):
        lines = lines[1:]
    df = pd.read_csv(pd.compat.StringIO(''.join(lines)))
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df

csv_path = os.path.join(os.path.dirname(__file__), "tradebook-JYD682-FO.csv")
df = load_trades(csv_path)

# --- Add strategy tagging column ---
if 'strategy' not in df.columns:
    df['strategy'] = ""

st.title("Trading Journal Dashboard")

# --- KPIs ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Trades", len(df))
with col2:
    st.metric("Net Quantity", int(df['quantity'].sum()))
with col3:
    st.metric("Unique Symbols", df['symbol'].nunique())

# --- Trade Log Table with Editable Strategy Tags ---
st.subheader("Trade Log")
edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "strategy": st.column_config.TextColumn("Strategy Tag", required=False)
    }
)

# --- Save tagged strategies (optional) ---
if st.button("Save Strategies"):
    edited_df.to_csv(csv_path, index=False)
    st.success("Strategies saved!")

# --- Basic Analytics ---
st.subheader("Analytics")
st.bar_chart(df.groupby('trade_date')['quantity'].sum())
