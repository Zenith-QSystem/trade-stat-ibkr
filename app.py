import streamlit as st
import pandas as pd
import numpy as np
import pytz

# ================= 核心业务逻辑 =================

def standardize_data(df: pd.DataFrame, data_tz: str, stats_tz: str) -> pd.DataFrame:
    """
    清洗并标准化数据，包含时区转换处理。
    """
    # 1. 基础映射清洗
    df.rename(columns=lambda x: str(x).strip(), inplace=True)
    col_mapping = {
        '商品代码': 'Symbol', '买/卖': 'Side', '数量': 'Qty',
        '执行价': 'Price', 'Fill Price': 'Price', '时间': 'Time',
        '净额': 'Net_Amount', 'Net Amount': 'Net_Amount', '手续费': 'Commission'
    }
    df.rename(columns=col_mapping, inplace=True)
    
    required_cols = ['Symbol', 'Side', 'Qty', 'Price', 'Time', 'Net_Amount', 'Commission']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失必要的列 / Missing columns: {missing_cols}")
    
    side_mapping = {'买入': 'Buy', '卖出': 'Sell', 'Buy': 'Buy', 'Sell': 'Sell'}
    df['Side'] = df['Side'].str.strip().str.title().map(side_mapping).fillna(df['Side'])
    
    # 2. 时区处理核心逻辑
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 如果原始时间数据没有时区信息 (Naive)，先赋予其【数据原时区】
    if df['Time'].dt.tz is None:
        df['Time'] = df['Time'].dt.tz_localize(data_tz)
    else:
        # 如果原始数据已经自带时区，则转换到【数据原时区】统一基准
        df['Time'] = df['Time'].dt.tz_convert(data_tz)
        
    # 将时间统一转换为【统计分割时区】
    df['Time'] = df['Time'].dt.tz_convert(stats_tz)
    
    # 3. 按统计时区的时间排序
    df = df.sort_values(by='Time').reset_index(drop=True)
    
    return df

def get_multiplier(df: pd.DataFrame) -> float:
    """计算合约乘数"""
    valid_rows = df[(df['Price'] > 0) & (df['Qty'] > 0)]
    if not valid_rows.empty:
        row = valid_rows.iloc[0]
        multiplier = round(row['Net_Amount'] / (row['Price'] * row['Qty']))
        return multiplier if multiplier > 0 else 1.0
    return 1.0

def calculate_fifo(df: pd.DataFrame) -> pd.DataFrame:
    """核心 FIFO 匹配引擎"""
    multiplier = get_multiplier(df)
    trades = []
    open_positions = {}

    for row in df.itertuples(index=False):
        sym, side, qty, price, time, _, comm = (
            row.Symbol, row.Side, row.Qty, row.Price, row.Time, row.Net_Amount, row.Commission
        )
        comm_per_qty = comm / qty
        inventory = open_positions.setdefault(sym, [])

        while qty > 0:
            if not inventory or inventory[0]['side'] == side:
                inventory.append({
                    'side': side, 'qty': qty, 'price': price, 
                    'comm_per_qty': comm_per_qty, 'time': time
                })
                qty = 0 
            else:
                match = inventory[0]
                match_qty = min(qty, match['qty'])

                buy_price = price if side == 'Buy' else match['price']
                sell_price = match['price'] if side == 'Buy' else price

                gross_pnl = (sell_price - buy_price) * multiplier * match_qty
                total_comm = (comm_per_qty * match_qty) + (match['comm_per_qty'] * match_qty)
                net_pnl = gross_pnl - total_comm

                trades.append({
                    'Symbol': sym,
                    'Open_Time': match['time'],
                    'Close_Time': time, # 此时的 time 已经是【统计时区】下的时间了
                    'Close_Side': side,
                    'Qty': match_qty,
                    'Buy_Price': buy_price,
                    'Sell_Price': sell_price,
                    'Gross_PnL': gross_pnl,
                    'Commission': total_comm,
                    'Net_Profit': net_pnl
                })

                qty -= match_qty
                match['qty'] -= match_qty

                if match['qty'] == 0:
                    inventory.pop(0)

    return pd.DataFrame(trades)

def compute_daily_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    """按日计算统计指标"""
    if trades_df.empty:
        return pd.DataFrame()

    # 因为 Close_Time 已经被转成了【统计时区】，这里的 date 就是该时区下正确的“一天”
    trades_df['Date'] = trades_df['Close_Time'].dt.date
    daily_stats = []

    for date, group in trades_df.groupby('Date'):
        total_trades = len(group)
        wins = group[group['Net_Profit'] > 0]
        losses = group[group['Net_Profit'] <= 0]

        net_profit = group['Net_Profit'].sum()
        gross_profit = wins['Net_Profit'].sum()
        gross_loss = abs(losses['Net_Profit'].sum())
        
        win_rate = len(wins) / total_trades if total_trades else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0)
        
        avg_win = gross_profit / len(wins) if len(wins) > 0 else 0
        avg_loss = gross_loss / len(losses) if len(losses) > 0 else 0
        avg_pnl_ratio = (avg_win / avg_loss) if avg_loss > 0 else (np.inf if avg_win > 0 else 0)

        daily_stats.append({
            'Date': date,
            'Net Profit': net_profit,
            'Win Rate (%)': win_rate * 100,
            'Profit Factor': profit_factor,
            'Avg PnL Ratio': avg_pnl_ratio,
            'Trades': total_trades
        })
        
    return pd.DataFrame(daily_stats)

# ================= UI 渲染逻辑 =================

st.set_page_config(page_title="Trade Analyzer | 交易记录分析", layout="wide")

# 侧边栏设置
st.sidebar.header("⚙️ 时区设置 / Timezone Settings")

# 预设常用的时区映射
TIMEZONE_OPTIONS = {
    "Asia/Shanghai (UTC+8 北京/亚洲)": "Asia/Shanghai",
    "America/New_York (EST/EDT 美东)": "America/New_York",
    "America/Chicago (CST/CDT 美中)": "America/Chicago",
    "America/Los_Angeles (PST/PDT 美西)": "America/Los_Angeles",
    "UTC (协调世界时)": "UTC",
    "Europe/London (GMT/BST 英国)": "Europe/London"
}

data_tz_key = st.sidebar.selectbox(
    "1. 数据源时区 / Data Timezone\n(CSV 文件中的时间属于哪个时区？)", 
    list(TIMEZONE_OPTIONS.keys()), 
    index=0 # 默认 "Asia/Shanghai"
)

stats_tz_key = st.sidebar.selectbox(
    "2. 统计分割时区 / Stats Timezone\n(你想以哪个时区的 00:00 作为一天？)", 
    list(TIMEZONE_OPTIONS.keys()), 
    index=1 # 默认 "America/New_York"
)

data_tz = TIMEZONE_OPTIONS[data_tz_key]
stats_tz = TIMEZONE_OPTIONS[stats_tz_key]

st.sidebar.divider()
st.sidebar.markdown(f"**当前转换逻辑:**\n- 原始数据被视为: `{data_tz}`\n- 按日统计按照: `{stats_tz}` 分割")

# 主界面
st.title("📈 交易记录分析器")
st.markdown("上传 CSV 文件（支持中/英文表头），系统将自动使用 **先进先出 (FIFO)** 原则匹配闭合交易，并生成每日数据报告。")

uploaded_file = st.file_uploader("拖拽或点击上传 CSV 文件 / Drag and drop CSV here", type=['csv'])

if uploaded_file:
    try:
        # 1. 加载与清洗 (带入时区参数)
        df_raw = pd.read_csv(uploaded_file)
        df_clean = standardize_data(df_raw, data_tz, stats_tz)

        with st.expander("预览解析后的源数据 (已转换至统计时区) / Raw Data in Stats Timezone"):
            st.dataframe(df_clean.head(10))

        with st.spinner('正在运行 FIFO 撮合引擎... / Running FIFO Engine...'):
            # 2. 计算 FIFO 与日统计
            trades_df = calculate_fifo(df_clean)
            
            if trades_df.empty:
                st.warning("⚠️ 数据中未发现闭合（已平仓）的交易记录！/ No closed trades found!")
                st.stop()
                
            daily_df = compute_daily_stats(trades_df)

            # 3. 渲染顶栏指标
            st.subheader("📊 整体表现 / Overall Performance")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            total_net = trades_df['Net_Profit'].sum()
            total_trades = len(trades_df)
            overall_win_rate = len(trades_df[trades_df['Net_Profit'] > 0]) / total_trades * 100
            
            gross_p = trades_df[trades_df['Net_Profit'] > 0]['Net_Profit'].sum()
            gross_l = abs(trades_df[trades_df['Net_Profit'] <= 0]['Net_Profit'].sum())
            overall_pf = gross_p / gross_l if gross_l > 0 else float('inf')

            metrics_col1.metric("总净利润 / Total Net Profit", f"{total_net:,.2f}")
            metrics_col2.metric("交易笔数 / Total Trades", total_trades)
            metrics_col3.metric("整体胜率 / Win Rate", f"{overall_win_rate:.1f}%")
            metrics_col4.metric("整体盈利因子 / Profit Factor", 
                                f"{overall_pf:.2f}" if overall_pf != float('inf') else "∞")

            st.divider()

            # 4. 图表与报表
            st.subheader(f"📅 每日统计 / Daily Report (Timezone: {stats_tz})")
            
            styled_daily_df = daily_df.style.format({
                'Net Profit': "{:,.2f}",
                'Win Rate (%)': "{:.2f}%",
                'Profit Factor': lambda x: "∞" if x == float('inf') else f"{x:.2f}",
                'Avg PnL Ratio': lambda x: "∞" if x == float('inf') else f"{x:.2f}",
            })
            st.dataframe(styled_daily_df, use_container_width=True)

            st.subheader("📈 每日净利走势 / Daily Net Profit Chart")
            st.bar_chart(daily_df.set_index('Date')['Net Profit'])

            # 5. FIFO 明细账
            with st.expander("🔍 查看 FIFO 匹配明细 / View FIFO Matched Trades"):
                # 转换 datetime 列的格式，以便在 Streamlit 中更优雅地展示
                display_trades = trades_df.copy()
                display_trades['Open_Time'] = display_trades['Open_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_trades['Close_Time'] = display_trades['Close_Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

                styled_trades_df = display_trades.style.format({
                    'Buy_Price': "{:.2f}",
                    'Sell_Price': "{:.2f}",
                    'Gross_PnL': "{:.2f}",
                    'Commission': "{:.2f}",
                    'Net_Profit': "{:.2f}",
                })
                st.dataframe(styled_trades_df, use_container_width=True)

    except ValueError as ve:
        st.error(f"数据格式错误: {ve}")
    except Exception as e:
        st.error(f"处理数据时发生未知错误 / An error occurred: {e}")