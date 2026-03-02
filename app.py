import streamlit as st
import pandas as pd
import numpy as np
import pytz
import re
import io

# ================= 数据解析与清洗层 =================

def detect_and_parse_csv(uploaded_file):
    """
    自动识别 CSV 格式并提取标准化的 DataFrame。
    返回: (df_clean, has_time)
    """
    # 读取为字符串列表进行分析
    content = uploaded_file.getvalue().decode("utf-8", errors='ignore')
    lines = content.splitlines()
    
    if not lines:
        raise ValueError("上传的文件为空 / The uploaded file is empty.")

    # 嗅探格式：如果第一行包含 "Statement" 或 "Transaction History"
    is_ibkr_format = "Statement" in lines[0] or "Transaction History" in lines[0] or "总结" in lines[0]

    if is_ibkr_format:
        return _parse_ibkr_format(lines)
    else:
        return _parse_standard_format(uploaded_file)

def _parse_ibkr_format(lines):
    """
    解析 IBKR (盈透) Statement 格式 (仅日期，无时间)
    """
    trade_data = []
    for line in lines:
        if line.startswith("Transaction History,Data,"):
            parts = line.split(",")
            # 标准列索引 (基于提供的样本):
            # [2]日期, [5]交易类型(买/卖), [6]代码, [7]数量, [8]价格, [10]总额(含括号), [11]佣金
            
            # 清理金额字段中的特殊字符如 "-342962.5(1)" -> 342962.5
            raw_amt = parts[10]
            amt_cleaned = re.sub(r'[^\d.-]', '', raw_amt)
            
            trade_data.append({
                'Symbol': parts[6].strip(),
                'Side': parts[5].strip(),
                'Qty': abs(float(parts[7])), # 取绝对值
                'Price': float(parts[8]),
                'Time': pd.to_datetime(parts[2].strip()), # 只有日期，时间默认为 00:00:00
                'Net_Amount': abs(float(amt_cleaned)),
                'Commission': abs(float(parts[11]))
            })
            
    df = pd.DataFrame(trade_data)
    
    if df.empty:
        raise ValueError("未在该格式中找到任何交易记录 / No trades found in this statement.")
        
    # 【关键处理】IBKR 的报表默认是自下而上（最新记录在最前）。
    # 因为没有时分秒，必须把整个列表彻底倒序，才能恢复真实的先后发生顺序。
    df = df.iloc[::-1].reset_index(drop=True)
    
    # 映射买卖方向
    side_mapping = {'买': 'Buy', '卖': 'Sell', 'Buy': 'Buy', 'Sell': 'Sell'}
    df['Side'] = df['Side'].str.title().map(side_mapping).fillna(df['Side'])
    
    # 因为已经物理倒序，这里使用 stable (mergesort) 保留同一天内的原始先后顺序
    df = df.sort_values(by='Time', kind='mergesort').reset_index(drop=True)
    
    return df, False  # False 表示没有具体时间

def _parse_standard_format(uploaded_file):
    """
    解析标准扁平表格格式 (包含具体时间)
    """
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
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
    
    df['Time'] = pd.to_datetime(df['Time'])
    
    # 按时间正序排列
    df = df.sort_values(by='Time').reset_index(drop=True)
    
    return df, True  # True 表示含有具体时间

def apply_timezone_if_needed(df, data_tz, stats_tz):
    """处理时区转换逻辑"""
    if df['Time'].dt.tz is None:
        df['Time'] = df['Time'].dt.tz_localize(data_tz)
    else:
        df['Time'] = df['Time'].dt.tz_convert(data_tz)
        
    df['Time'] = df['Time'].dt.tz_convert(stats_tz)
    df = df.sort_values(by='Time', kind='mergesort').reset_index(drop=True)
    return df

# ================= 业务计算层 (FIFO & Stats) =================

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
                    'Close_Time': time,
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
    if trades_df.empty: return pd.DataFrame()

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

# ================= UI 渲染层 =================

st.set_page_config(page_title="Trade Analyzer | 交易记录分析", layout="wide")

st.title("📈 交易记录分析器")
st.markdown("支持 **tradingview导出的IBKR交易历史记录CSV** 与 **IBKR网页上导出的交易历史CSV格式**。")

uploaded_file = st.file_uploader("拖拽或点击上传 CSV 文件 / Drag and drop CSV here", type=['csv'])

if uploaded_file:
    try:
        # 1. 解析数据，并感知格式
        df_clean, has_time = detect_and_parse_csv(uploaded_file)
        
        # 2. 时区面板逻辑：只有在包含时分秒的格式下才显示，如果只有日期则跳过
        if has_time:
            st.sidebar.header("⚙️ 时区设置 / Timezone Settings")
            TIMEZONE_OPTIONS = {
                "Asia/Shanghai (UTC+8 北京/亚洲)": "Asia/Shanghai",
                "America/New_York (EST/EDT 美东)": "America/New_York",
                "America/Chicago (CST/CDT 美中)": "America/Chicago",
                "America/Los_Angeles (PST/PDT 美西)": "America/Los_Angeles",
                "UTC (协调世界时)": "UTC",
                "Europe/London (GMT/BST 英国)": "Europe/London"
            }
            data_tz_key = st.sidebar.selectbox("1. 数据源时区 (Data Timezone)", list(TIMEZONE_OPTIONS.keys()), index=0)
            stats_tz_key = st.sidebar.selectbox("2. 统计分割时区 (Stats Timezone)", list(TIMEZONE_OPTIONS.keys()), index=1)
            
            # 应用时区转换
            df_clean = apply_timezone_if_needed(df_clean, TIMEZONE_OPTIONS[data_tz_key], TIMEZONE_OPTIONS[stats_tz_key])
            st.sidebar.success("已启用时区转换。")
        else:
            st.sidebar.info("💡 格式感知：\n检测到该 CSV 为**仅包含日期的报表格式** (如 IBKR Statement)。已自动跳过时区转换环节。")

        # 3. 运行 FIFO 及数据展示
        with st.expander("预览底层标准化后的源数据 / Raw Data Standardized"):
            # 根据是否带时间，灵活格式化展示
            display_df = df_clean.copy()
            time_format = '%Y-%m-%d %H:%M:%S' if has_time else '%Y-%m-%d'
            display_df['Time'] = display_df['Time'].dt.strftime(time_format)
            st.dataframe(display_df.head(10))

        with st.spinner('正在运行 FIFO 撮合引擎... / Running FIFO Engine...'):
            trades_df = calculate_fifo(df_clean)
            
            if trades_df.empty:
                st.warning("⚠️ 数据中未发现闭合（已平仓）的交易记录！/ No closed trades found!")
                st.stop()
                
            daily_df = compute_daily_stats(trades_df)

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
            metrics_col4.metric("整体盈利因子 / Profit Factor", f"{overall_pf:.2f}" if overall_pf != float('inf') else "∞")

            st.divider()

            st.subheader(f"📅 每日统计 / Daily Report")
            styled_daily_df = daily_df.style.format({
                'Net Profit': "{:,.2f}", 'Win Rate (%)': "{:.2f}%",
                'Profit Factor': lambda x: "∞" if x == float('inf') else f"{x:.2f}",
                'Avg PnL Ratio': lambda x: "∞" if x == float('inf') else f"{x:.2f}",
            })
            st.dataframe(styled_daily_df, use_container_width=True)

            st.subheader("📈 每日净利走势 / Daily Net Profit Chart")
            st.bar_chart(daily_df.set_index('Date')['Net Profit'])

            with st.expander("🔍 查看 FIFO 匹配明细 / View FIFO Matched Trades"):
                display_trades = trades_df.copy()
                display_trades['Open_Time'] = display_trades['Open_Time'].dt.strftime(time_format)
                display_trades['Close_Time'] = display_trades['Close_Time'].dt.strftime(time_format)

                styled_trades_df = display_trades.style.format({
                    'Buy_Price': "{:.2f}", 'Sell_Price': "{:.2f}",
                    'Gross_PnL': "{:.2f}", 'Commission': "{:.2f}", 'Net_Profit': "{:.2f}",
                })
                st.dataframe(styled_trades_df, use_container_width=True)

    except ValueError as ve:
        st.error(f"解析出错: {ve}")
    except Exception as e:
        st.error(f"处理数据时发生未知错误 / An error occurred: {e}")