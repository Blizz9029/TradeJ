import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Trading Journal", layout="wide")

def load_trades(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    if 'strategy' not in df.columns:
        df['strategy'] = ""
    return df

st.title("Trading Journal Dashboard")

uploaded_file = st.file_uploader("Upload your tradebook CSV", type="csv")
if uploaded_file is not None:
    df = load_trades(uploaded_file)

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", len(df))
    with col2:
        st.metric("Net Quantity", int(df['quantity'].sum()))
    with col3:
        st.metric("Unique Symbols", df['symbol'].nunique())

    # Trade Log Table with Editable Strategy Tags
    st.subheader("Trade Log")
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "strategy": st.column_config.TextColumn("Strategy Tag", required=False)
        }
    )

    # Save tagged strategies (optional)
    if st.button("Save Strategies"):
        edited_df.to_csv("tagged_tradebook.csv", index=False)
        st.success("Strategies saved to tagged_tradebook.csv!")

    # Basic Analytics
    st.subheader("Analytics")
    st.bar_chart(df.groupby('trade_date')['quantity'].sum())
else:
    st.info("Please upload your tradebook CSV file to get started.")
