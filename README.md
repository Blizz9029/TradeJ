# Trade Journal - Streamlit Application

This is a Streamlit-based trade journal application that allows traders to upload CSV files of their trades, tag them with strategies, and analyze performance.

## Features

- **Dashboard**: Overview of trading performance with key metrics
- **Upload Trades**: Import trades from CSV files
- **Trade Log**: View and tag trades with strategies
- **Strategies**: Create and manage trading strategies
- **Calendar**: View trading activity on a calendar
- **Journal**: Daily trading journal with notes

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

## CSV Format

Your CSV file should include the following columns:
- `symbol`: Trading symbol
- `trade_date`: Date of the trade (YYYY-MM-DD format)
- `trade_type`: Type of trade (buy or sell)
- `quantity`: Number of shares/contracts
- `price`: Price per share/contract

Additional columns like isin, exchange, segment, series, etc. will also be processed if present.

## Strategy Tagging

The application allows you to tag trades with different strategies to analyze performance by strategy. You can:
1. Create custom strategies with names, descriptions, and colors
2. Tag trades with strategies in the Trade Log
3. View performance metrics by strategy in the Dashboard

## Data Storage

All data is stored locally in a SQLite database (`trade_journal.db`).
