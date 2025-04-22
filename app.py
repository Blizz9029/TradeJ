import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
from datetime import datetime
import uuid

# Set page configuration
st.set_page_config(
    page_title="Trade Journal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_db():
    conn = sqlite3.connect('trade_journal.db')
    c = conn.cursor()
    
    # Create trades table
    c.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        isin TEXT,
        trade_date TEXT NOT NULL,
        exchange TEXT,
        segment TEXT,
        series TEXT,
        trade_type TEXT NOT NULL,
        auction BOOLEAN,
        quantity REAL NOT NULL,
        price REAL NOT NULL,
        trade_id TEXT,
        order_id TEXT,
        order_execution_time TEXT,
        expiry_date TEXT,
        strategy_id TEXT
    )
    ''')
    
    # Create strategies table
    c.execute('''
    CREATE TABLE IF NOT EXISTS strategies (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        color TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create trade_notes table
    c.execute('''
    CREATE TABLE IF NOT EXISTS trade_notes (
        id TEXT PRIMARY KEY,
        trade_id TEXT NOT NULL,
        note TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (trade_id) REFERENCES trades(id)
    )
    ''')
    
    # Insert default strategies if they don't exist
    default_strategies = [
        ('1', 'Momentum', 'Trading based on price momentum', '#4CAF50'),
        ('2', 'Breakout', 'Trading breakouts from consolidation', '#2196F3'),
        ('3', 'Reversal', 'Trading price reversals', '#F44336'),
        ('4', 'Swing', 'Multi-day position trading', '#9C27B0'),
        ('5', 'Scalping', 'Quick in-and-out trades', '#FF9800')
    ]
    
    for strategy in default_strategies:
        c.execute('INSERT OR IGNORE INTO strategies (id, name, description, color) VALUES (?, ?, ?, ?)', strategy)
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Helper functions
def format_currency(amount):
    return f"₹{amount:,.2f}"

def calculate_pnl(trades_df):
    if trades_df.empty:
        return 0
    
    buy_value = trades_df[trades_df['trade_type'] == 'buy']['quantity'] * trades_df[trades_df['trade_type'] == 'buy']['price']
    sell_value = trades_df[trades_df['trade_type'] == 'sell']['quantity'] * trades_df[trades_df['trade_type'] == 'sell']['price']
    
    return sell_value.sum() - buy_value.sum()

def calculate_win_rate(trades_df):
    if trades_df.empty:
        return 0
    
    symbols = trades_df['symbol'].unique()
    wins = 0
    
    for symbol in symbols:
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        pnl = calculate_pnl(symbol_trades)
        if pnl > 0:
            wins += 1
    
    return (wins / len(symbols)) * 100 if len(symbols) > 0 else 0

def calculate_profit_factor(trades_df):
    if trades_df.empty:
        return 0
    
    symbols = trades_df['symbol'].unique()
    gross_profit = 0
    gross_loss = 0
    
    for symbol in symbols:
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        pnl = calculate_pnl(symbol_trades)
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)
    
    return gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

def get_all_strategies():
    conn = sqlite3.connect('trade_journal.db')
    strategies = pd.read_sql('SELECT * FROM strategies ORDER BY name', conn)
    conn.close()
    return strategies

def get_all_trades():
    conn = sqlite3.connect('trade_journal.db')
    trades = pd.read_sql('SELECT * FROM trades ORDER BY trade_date DESC', conn)
    conn.close()
    return trades

def get_trades_by_strategy(strategy_id):
    conn = sqlite3.connect('trade_journal.db')
    trades = pd.read_sql('SELECT * FROM trades WHERE strategy_id = ? ORDER BY trade_date DESC', conn, params=(strategy_id,))
    conn.close()
    return trades

def get_trades_by_date(date):
    conn = sqlite3.connect('trade_journal.db')
    trades = pd.read_sql('SELECT * FROM trades WHERE date(trade_date) = ? ORDER BY trade_date DESC', conn, params=(date,))
    conn.close()
    return trades

def save_trades(df):
    conn = sqlite3.connect('trade_journal.db')
    c = conn.cursor()
    
    for _, row in df.iterrows():
        # Generate a unique ID if not present
        trade_id = row.get('id', str(uuid.uuid4()))
        
        # Check if trade already exists
        c.execute('SELECT id FROM trades WHERE trade_id = ?', (row.get('trade_id', ''),))
        existing = c.fetchone()
        
        if existing:
            continue  # Skip if trade already exists
        
        # Insert new trade
        c.execute('''
        INSERT INTO trades (
            id, symbol, isin, trade_date, exchange, segment, series, 
            trade_type, auction, quantity, price, trade_id, 
            order_id, order_execution_time, expiry_date, strategy_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id,
            row['symbol'],
            row.get('isin', None),
            row['trade_date'],
            row.get('exchange', None),
            row.get('segment', None),
            row.get('series', None),
            row['trade_type'],
            row.get('auction', False),
            row['quantity'],
            row['price'],
            row.get('trade_id', None),
            row.get('order_id', None),
            row.get('order_execution_time', None),
            row.get('expiry_date', None),
            None  # No strategy assigned initially
        ))
    
    conn.commit()
    conn.close()

def update_trade_strategy(trade_id, strategy_id):
    conn = sqlite3.connect('trade_journal.db')
    c = conn.cursor()
    c.execute('UPDATE trades SET strategy_id = ? WHERE id = ?', (strategy_id, trade_id))
    conn.commit()
    conn.close()

def create_strategy(name, description, color):
    conn = sqlite3.connect('trade_journal.db')
    c = conn.cursor()
    strategy_id = str(uuid.uuid4())
    c.execute(
        'INSERT INTO strategies (id, name, description, color) VALUES (?, ?, ?, ?)',
        (strategy_id, name, description, color)
    )
    conn.commit()
    conn.close()
    return strategy_id

# Sidebar navigation
st.sidebar.title("Trade Journal")
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Upload Trades", "Trade Log", "Strategies", "Calendar", "Journal"]
)

# Get data for all pages
all_trades = get_all_trades()
all_strategies = get_all_strategies()

# Dashboard page
if page == "Dashboard":
    st.title("Dashboard")
    
    if all_trades.empty:
        st.info("No trades found. Upload a CSV file to get started.")
    else:
        # Calculate overall metrics
        total_pnl = calculate_pnl(all_trades)
        win_rate = calculate_win_rate(all_trades)
        profit_factor = calculate_profit_factor(all_trades)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Net P&L", format_currency(total_pnl))
        with col2:
            st.metric("Win Rate", f"{win_rate:.2f}%")
        with col3:
            st.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞")
        with col4:
            st.metric("Total Trades", len(all_trades))
        
        # P&L by date chart
        st.subheader("Daily P&L")
        
        # Group trades by date and calculate P&L
        all_trades['date'] = pd.to_datetime(all_trades['trade_date']).dt.date
        daily_pnl = all_trades.groupby('date').apply(calculate_pnl).reset_index()
        daily_pnl.columns = ['date', 'pnl']
        
        # Create P&L chart
        fig = px.bar(
            daily_pnl,
            x='date',
            y='pnl',
            color=daily_pnl['pnl'] > 0,
            color_discrete_map={True: 'green', False: 'red'},
            labels={'date': 'Date', 'pnl': 'P&L'},
            title='Daily P&L'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Strategy performance
        st.subheader("Strategy Performance")
        
        # Calculate performance by strategy
        strategy_performance = []
        
        # Add untagged trades
        untagged_trades = all_trades[all_trades['strategy_id'].isnull()]
        if not untagged_trades.empty:
            strategy_performance.append({
                'id': 'untagged',
                'name': 'Untagged',
                'color': '#CBD5E0',
                'pnl': calculate_pnl(untagged_trades),
                'win_rate': calculate_win_rate(untagged_trades),
                'profit_factor': calculate_profit_factor(untagged_trades),
                'trade_count': len(untagged_trades)
            })
        
        # Add strategy trades
        for _, strategy in all_strategies.iterrows():
            strategy_trades = all_trades[all_trades['strategy_id'] == strategy['id']]
            if not strategy_trades.empty:
                strategy_performance.append({
                    'id': strategy['id'],
                    'name': strategy['name'],
                    'color': strategy['color'],
                    'pnl': calculate_pnl(strategy_trades),
                    'win_rate': calculate_win_rate(strategy_trades),
                    'profit_factor': calculate_profit_factor(strategy_trades),
                    'trade_count': len(strategy_trades)
                })
        
        # Create DataFrame for display
        if strategy_performance:
            strategy_df = pd.DataFrame(strategy_performance)
            strategy_df = strategy_df.sort_values('pnl', ascending=False)
            
            # Format columns for display
            strategy_df['pnl'] = strategy_df['pnl'].apply(format_currency)
            strategy_df['win_rate'] = strategy_df['win_rate'].apply(lambda x: f"{x:.2f}%")
            strategy_df['profit_factor'] = strategy_df['profit_factor'].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
            
            # Display strategy performance table
            st.dataframe(
                strategy_df[['name', 'pnl', 'win_rate', 'profit_factor', 'trade_count']],
                column_config={
                    'name': 'Strategy',
                    'pnl': 'P&L',
                    'win_rate': 'Win Rate',
                    'profit_factor': 'Profit Factor',
                    'trade_count': 'Trades'
                },
                use_container_width=True
            )
        else:
            st.info("No strategy performance data available. Tag your trades with strategies to see performance metrics.")
        
        # Recent trades
        st.subheader("Recent Trades")
        recent_trades = all_trades.head(10).copy()
        
        # Format columns for display
        recent_trades['trade_date'] = pd.to_datetime(recent_trades['trade_date']).dt.strftime('%Y-%m-%d')
        recent_trades['price'] = recent_trades['price'].apply(format_currency)
        recent_trades['value'] = (recent_trades['quantity'] * pd.to_numeric(recent_trades['price'].str.replace('₹', '').str.replace(',', ''))).apply(format_currency)
        
        # Add strategy names
        strategy_map = dict(zip(all_strategies['id'], all_strategies['name']))
        recent_trades['strategy'] = recent_trades['strategy_id'].map(lambda x: strategy_map.get(x, 'Untagged'))
        
        # Display recent trades table
        st.dataframe(
            recent_trades[['trade_date', 'symbol', 'trade_type', 'quantity', 'price', 'value', 'strategy']],
            column_config={
                'trade_date': 'Date',
                'symbol': 'Symbol',
                'trade_type': 'Type',
                'quantity': 'Quantity',
                'price': 'Price',
                'value': 'Value',
                'strategy': 'Strategy'
            },
            use_container_width=True
        )

# Upload Trades page
elif page == "Upload Trades":
    st.title("Upload Trades")
    
    with st.container():
        st.write("Upload your trade CSV file to analyze your trading performance.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ['symbol', 'trade_date', 'trade_type', 'quantity', 'price']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Display preview
                    st.subheader("Preview")
                    st.dataframe(df.head())
                    
                    # Process and save trades
                    if st.button("Import Trades"):
                        with st.spinner("Importing trades..."):
                            save_trades(df)
                            st.success(f"Successfully imported {len(df)} trades.")
                            st.rerun()  # Refresh the page to show updated data
            
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
        
        # CSV format information
        st.subheader("CSV Format Requirements")
        st.write("Your CSV file should include the following columns:")
        
        format_df = pd.DataFrame({
            'Column': ['symbol', 'trade_date', 'trade_type', 'quantity', 'price'],
            'Description': [
                'Trading symbol',
                'Date of the trade (YYYY-MM-DD format)',
                'Type of trade (buy or sell)',
                'Number of shares/contracts',
                'Price per share/contract'
            ],
            'Example': ['ICICIBANK', '2025-04-17', 'buy', '700', '1385.80']
        })
        
        st.dataframe(format_df, use_container_width=True)
        
        st.write("Additional columns like isin, exchange, segment, series, etc. will also be processed if present.")

# Trade Log page
elif page == "Trade Log":
    st.title("Trade Log")
    
    if all_trades.empty:
        st.info("No trades found. Upload a CSV file to get started.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols = ['All'] + sorted(all_trades['symbol'].unique().tolist())
            selected_symbol = st.selectbox("Symbol", symbols)
        
        with col2:
            strategies_list = [{'id': 'all', 'name': 'All'}] + \
                             [{'id': 'untagged', 'name': 'Untagged'}] + \
                             all_strategies.to_dict('records')
            selected_strategy = st.selectbox(
                "Strategy",
                options=[s['id'] for s in strategies_list],
                format_func=lambda x: next((s['name'] for s in strategies_list if s['id'] == x), 'Unknown')
            )
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(
                    pd.to_datetime(all_trades['trade_date']).min().date(),
                    pd.to_datetime(all_trades['trade_date']).max().date()
                ),
                max_value=datetime.now().date()
            )
        
        # Apply filters
        filtered_trades = all_trades.copy()
        
        if selected_symbol != 'All':
            filtered_trades = filtered_trades[filtered_trades['symbol'] == selected_symbol]
        
        if selected_strategy != 'all':
            if selected_strategy == 'untagged':
                filtered_trades = filtered_trades[filtered_trades['strategy_id'].isnull()]
            else:
                filtered_trades = filtered_trades[filtered_trades['strategy_id'] == selected_strategy]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_trades = filtered_trades[
                (pd.to_datetime(filtered_trades['trade_date']).dt.date >= start_date) &
                (pd.to_datetime(filtered_trades['trade_date']).dt.date <= end_date)
            ]
        
        # Strategy tagging
        if not filtered_trades.empty:
            st.subheader("Tag Trades with Strategy")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                tag_strategy = st.selectbox(
                    "Select Strategy",
                    options=[''] + [s['id'] for s in all_strategies.to_dict('records')],
                    format_func=lambda x: '' if x == '' else next((s['name'] for s in all_strategies.to_dict('records') if s['id'] == x), 'Unknown')
                )
            
            with col2:
                tag_button = st.button("Tag Selected Trades")
            
            # Display trades table with selection
            st.subheader("Trades")
            
            # Format trades for display
            display_trades = filtered_trades.copy()
            display_trades['trade_date'] = pd.to_datetime(display_trades['trade_date']).dt.strftime('%Y-%m-%d')
            display_trades['price'] = display_trades['price'].apply(format_currency)
            display_trades['value'] = (display_trades['quantity'] * pd.to_numeric(display_trades['price'].str.replace('₹', '').str.replace(',', ''))).apply(format_currency)
            
            # Add strategy names
            strategy_map = dict(zip(all_strategies['id'], all_strategies['name']))
            display_trades['strategy'] = display_trades['strategy_id'].map(lambda x: strategy_map.get(x, 'Untagged'))
            
            # Add selection column
            selected_trades = st.multiselect(
                "Select trades to tag",
                options=display_trades['id'].tolist(),
                format_func=lambda x: f"{display_trades[display_trades['id'] == x]['symbol'].values[0]} - {display_trades[display_trades['id'] == x]['trade_date'].values[0]} - {display_trades[display_trades['id'] == x]['trade_type'].values[0]}"
            )
            
            # Tag selected trades
            if tag_button and selected_trades and tag_strategy:
                with st.spinner("Tagging trades..."):
                    for trade_id in selected_trades:
                        update_trade_strategy(trade_id, tag_strategy)
                    st.success(f"Successfully tagged {len(selected_trades)} trades.")
                    st.rerun()  # Refresh the page to show updated data
            
            # Display trades table
            st.dataframe(
                display_trades[['trade_date', 'symbol', 'trade_type', 'quantity', 'price', 'value', 'strategy']],
                column_config={
                    'trade_date': 'Date',
                    'symbol': 'Symbol',
                    'trade_type': 'Type',
                    'quantity': 'Quantity',
                    'price': 'Price',
                    'value': 'Value',
                    'strategy': 'Strategy'
                },
                use_container_width=True
            )
        else:
            st.info("No trades match the selected filters.")

# Strategies page
elif page == "Strategies":
    st.title("Trading Strategies")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Create New Strategy")
        
        with st.form("create_strategy_form"):
            strategy_name = st.text_input("Strategy Name", placeholder="e.g., Momentum Breakout")
            strategy_description = st.text_area("Description", placeholder="Describe your strategy...")
            strategy_color = st.color_picker("Color", "#4CAF50")
            
            submit_button = st.form_submit_button("Create Strategy")
            
            if submit_button:
                if not strategy_name:
                    st.error("Strategy name is required")
                else:
                    create_strategy(strategy_name, strategy_description, strategy_color)
                    st.success(f"Strategy '{strategy_name}' created successfully.")
                    st.rerun()  # Refresh the page to show the new strategy
    
    with col2:
        st.subheader("Your Strategies")
        
        if all_strategies.empty:
            st.info("No strategies found. Create your first strategy to get started.")
        else:
            # Count trades by strategy
            strategy_counts = {}
            for _, strategy in all_strategies.iterrows():
                count = len(all_trades[all_trades['strategy_id'] == strategy['id']])
                strategy_counts[strategy['id']] = count
            
            # Add trade counts to strategies dataframe
            all_strategies['trade_count'] = all_strategies['id'].map(strategy_counts)
            
            # Display strategies
            for _, strategy in all_strategies.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([0.1, 2, 0.5])
                    
                    with col1:
                        st.markdown(
                            f"<div style='width:20px;height:20px;border-radius:50%;background-color:{strategy['color']}'></div>",
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(f"**{strategy['name']}**")
                        if strategy['description']:
                            st.markdown(f"<small>{strategy['description']}</small>", unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"{strategy['trade_count']} trades")
                    
                    st.markdown("---")

# Calendar page
elif page == "Calendar":
    st.title("Trading Calendar")
    
    if all_trades.empty:
        st.info("No trades found. Upload a CSV file to get started.")
    else:
        # Date selection
        current_month = st.date_input(
            "Select Month",
            value=datetime.now().date().replace(day=1),
            format="YYYY-MM"
        )
        
        # Calculate start and end of month
        year = current_month.year
        month = current_month.month
        
        # Get all dates in the month
        import calendar
        _, num_days = calendar.monthrange(year, month)
        month_dates = [datetime(year, month, day).date() for day in range(1, num_days + 1)]
        
        # Get trading days in the month
        all_trades['date'] = pd.to_datetime(all_trades['trade_date']).dt.date
        trading_days = all_trades['date'].unique()
        
        # Calculate daily P&L
        daily_pnl = {}
        daily_trades = {}
        daily_win_rate = {}
        
        for date in trading_days:
            date_trades = all_trades[all_trades['date'] == date]
            daily_pnl[date] = calculate_pnl(date_trades)
            daily_trades[date] = len(date_trades)
            daily_win_rate[date] = calculate_win_rate(date_trades)
        
        # Create calendar grid
        st.subheader(current_month.strftime("%B %Y"))
        
        # Create week rows
        weeks = []
        week = []
        
        # Add empty cells for days before the 1st of the month
        first_day_weekday = month_dates[0].weekday()
        for _ in range(first_day_weekday):
            week.append(None)
        
        # Add all days of the month
        for date in month_dates:
            week.append(date)
            if len(week) == 7:
                weeks.append(week)
                week = []
        
        # Add empty cells for days after the last day of the month
        if week:
            while len(week) < 7:
                week.append(None)
            weeks.append(week)
        
        # Display calendar
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        # Create header row
        header_cols = st.columns(7)
        for i, day in enumerate(weekdays):
            with header_cols[i]:
                st.markdown(f"<div style='text-align:center;font-weight:bold'>{day}</div>", unsafe_allow_html=True)
        
        # Create calendar cells
        for week in weeks:
            cols = st.columns(7)
            for i, date in enumerate(week):
                with cols[i]:
                    if date:
                        # Check if it's a trading day
                        if date in trading_days:
                            pnl = daily_pnl.get(date, 0)
                            trade_count = daily_trades.get(date, 0)
                            win_rate = daily_win_rate.get(date, 0)
                            
                            # Create cell with trading data
                            st.markdown(
                                f"""
                                <div style='border:1px solid #ddd;border-radius:5px;padding:10px;height:100px;'>
                                    <div style='text-align:right;font-size:0.8em;color:#666'>{date.day}</div>
                                    <div style='color:{"green" if pnl >= 0 else "red"};font-weight:bold'>{format_currency(pnl)}</div>
                                    <div style='font-size:0.8em'>{trade_count} trade{'s' if trade_count != 1 else ''}</div>
                                    <div style='font-size:0.8em'>{win_rate:.0f}% win rate</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            # Create empty cell
                            st.markdown(
                                f"""
                                <div style='border:1px solid #ddd;border-radius:5px;padding:10px;height:100px;'>
                                    <div style='text-align:right;font-size:0.8em;color:#666'>{date.day}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
        
        # Monthly summary
        st.subheader("Monthly Summary")
        
        # Filter trades for the selected month
        month_start = datetime(year, month, 1).date()
        month_end = month_dates[-1]
        month_trades = all_trades[(all_trades['date'] >= month_start) & (all_trades['date'] <= month_end)]
        
        # Calculate monthly metrics
        monthly_pnl = calculate_pnl(month_trades)
        monthly_win_rate = calculate_win_rate(month_trades)
        monthly_profit_factor = calculate_profit_factor(month_trades)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Monthly P&L", format_currency(monthly_pnl))
        
        with col2:
            st.metric("Win Rate", f"{monthly_win_rate:.2f}%")
        
        with col3:
            st.metric("Profit Factor", f"{monthly_profit_factor:.2f}" if monthly_profit_factor != float('inf') else "∞")
        
        with col4:
            st.metric("Trading Days", len(set(month_trades['date'])))

# Journal page
elif page == "Journal":
    st.title("Daily Journal")
    
    if all_trades.empty:
        st.info("No trades found. Upload a CSV file to get started.")
    else:
        # Get unique trading dates
        all_trades['date'] = pd.to_datetime(all_trades['trade_date']).dt.date
        trading_dates = sorted(all_trades['date'].unique(), reverse=True)
        
        # Date selection
        selected_date = st.selectbox(
            "Select Date",
            options=trading_dates,
            format_func=lambda x: x.strftime("%A, %B %d, %Y")
        )
        
        if selected_date:
            # Get trades for selected date
            date_trades = all_trades[all_trades['date'] == selected_date]
            
            # Calculate date statistics
            date_pnl = calculate_pnl(date_trades)
            
            # Count winners and losers
            symbols = date_trades['symbol'].unique()
            winners = 0
            losers = 0
            
            for symbol in symbols:
                symbol_trades = date_trades[date_trades['symbol'] == symbol]
                symbol_pnl = calculate_pnl(symbol_trades)
                if symbol_pnl > 0:
                    winners += 1
                elif symbol_pnl < 0:
                    losers += 1
            
            # Calculate volume and profit factor
            volume = (date_trades['quantity'] * date_trades['price']).sum()
            profit_factor = calculate_profit_factor(date_trades)
            
            # Display date header with P&L
            st.subheader(f"{selected_date.strftime('%A, %B %d, %Y')} - {format_currency(date_pnl)}")
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", len(date_trades))
            
            with col2:
                st.metric("Winners/Losers", f"{winners}/{losers}")
            
            with col3:
                win_rate = (winners / len(symbols)) * 100 if len(symbols) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.2f}%")
            
            with col4:
                st.metric("Volume", format_currency(volume))
            
            # P&L chart
            st.subheader("P&L Chart")
            
            # Create cumulative P&L chart
            trades_by_time = date_trades.sort_values('trade_date')
            
            # Calculate running P&L
            running_pnl = []
            current_pnl = 0
            
            for _, trade in trades_by_time.iterrows():
                if trade['trade_type'] == 'buy':
                    current_pnl -= trade['quantity'] * trade['price']
                else:  # sell
                    current_pnl += trade['quantity'] * trade['price']
                running_pnl.append(current_pnl)
            
            # Create chart
            if running_pnl:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(running_pnl))),
                    y=running_pnl,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='green' if running_pnl[-1] >= 0 else 'red'),
                    fillcolor=f"rgba(0, 128, 0, 0.2)" if running_pnl[-1] >= 0 else f"rgba(255, 0, 0, 0.2)"
                ))
                fig.update_layout(
                    title="Cumulative P&L",
                    xaxis_title="Trades",
                    yaxis_title="P&L",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Trades table
            st.subheader("Trades")
            
            # Format trades for display
            display_trades = date_trades.copy()
            display_trades['trade_time'] = pd.to_datetime(display_trades['trade_date']).dt.strftime('%H:%M:%S')
            display_trades['price'] = display_trades['price'].apply(format_currency)
            display_trades['value'] = (display_trades['quantity'] * pd.to_numeric(display_trades['price'].str.replace('₹', '').str.replace(',', ''))).apply(format_currency)
            
            # Add strategy names
            strategy_map = dict(zip(all_strategies['id'], all_strategies['name']))
            display_trades['strategy'] = display_trades['strategy_id'].map(lambda x: strategy_map.get(x, 'Untagged'))
            
            # Display trades table
            st.dataframe(
                display_trades[['trade_time', 'symbol', 'trade_type', 'quantity', 'price', 'value', 'strategy']],
                column_config={
                    'trade_time': 'Time',
                    'symbol': 'Symbol',
                    'trade_type': 'Type',
                    'quantity': 'Quantity',
                    'price': 'Price',
                    'value': 'Value',
                    'strategy': 'Strategy'
                },
                use_container_width=True
            )
            
            # Journal notes
            st.subheader("Journal Notes")
            
            notes = st.text_area(
                "Write your trading notes for this day...",
                height=200
            )
            
            if st.button("Save Notes"):
                st.success("Notes saved successfully!")

# Add custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .st-emotion-cache-16txtl3 h1 {
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .st-emotion-cache-16txtl3 h2 {
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .st-emotion-cache-16txtl3 h3 {
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)
