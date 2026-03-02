import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import csv
import io
from datetime import timedelta

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(page_title="交易日志与盈亏日历分析", page_icon="📈", layout="wide")

# ==========================================
# 核心业务逻辑函数
# ==========================================
@st.cache_data(show_spinner="正在解析和清洗数据...")
def parse_and_clean_data(file_content: str) -> pd.DataFrame:
    """解析 IBKR 的 CSV 报表，提取交易历史记录并清洗"""
    reader = csv.reader(io.StringIO(file_content))
    header = None
    data_rows = []
    
    for row in reader:
        if not row:
            continue
        if row[0] == "Transaction History" and row[1] == "Header":
            header = row
        elif row[0] == "Transaction History" and row[1] == "Data":
            data_rows.append(row)
            
    if not header or not data_rows:
        return pd.DataFrame()
        
    df = pd.DataFrame(data_rows, columns=header)
    
    # 基础过滤：去除出入金、无代码的行
    df = df[df['代码'].notna() & (df['代码'] != '') & (df['代码'] != '-')]
    
    # 类型转换
    df['日期'] = pd.to_datetime(df['日期'])
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce')
    df['价格'] = pd.to_numeric(df['价格'], errors='coerce')
    df['佣金'] = pd.to_numeric(df['佣金'], errors='coerce').fillna(0)
    
    # 剔除无效数量
    df = df.dropna(subset=['数量'])
    df = df[df['数量'] != 0]
    
    # 清洗总额 (去除括号如 "343050.0(1)")
    df['总额'] = df['总额'].astype(str).str.replace(r'\(.*?\)', '', regex=True)
    df['总额'] = pd.to_numeric(df['总额'], errors='coerce')
    
    # ================= 极其重要：按时间正序排列（从旧到新） =================
    # IBKR 报表默认是从新到旧，必须反转才能正确进行 FIFO 匹配
    df = df.sort_values('日期', ascending=True).reset_index(drop=True)
    
    # 动态计算合约乘数：乘数 = |总额| / (|数量| * 价格)
    df['Multiplier'] = (df['总额'].abs() / (df['数量'].abs() * df['价格'])).round(0)
    df['Multiplier'] = df['Multiplier'].fillna(1.0).replace(0, 1.0)
    
    return df

@st.cache_data(show_spinner="正在进行 FIFO 开平仓匹配...")
def match_trades_fifo(df: pd.DataFrame) -> pd.DataFrame:
    """使用 FIFO 算法配对开仓和平仓"""
    open_positions = {}
    closed_trades = []
    
    for _, row in df.iterrows():
        sym = row['代码']
        date = row['日期']
        qty = row['数量']
        price = row['价格']
        comm = row['佣金']
        mult = row['Multiplier']
        
        if sym not in open_positions:
            open_positions[sym] = []
            
        trade_sign = np.sign(qty)
        
        # 当此笔交易还有未分配完的数量时循环
        while abs(qty) > 1e-6:
            # 如果当前没有持仓，或者持仓方向与当前交易同向 (加仓/开仓)
            if not open_positions[sym] or np.sign(open_positions[sym][0]['qty']) == trade_sign:
                open_positions[sym].append({
                    'date': date,
                    'qty': qty,
                    'price': price,
                    'comm': comm,
                    'mult': mult
                })
                qty = 0  # 数量已全部作为开仓入队
            else:
                # 存在反向持仓 -> 平仓匹配
                open_pos = open_positions[sym][0]
                open_qty = open_pos['qty']
                open_sign = np.sign(open_qty)
                
                # 本次匹配的数量取两者绝对值的较小者
                match_qty = min(abs(qty), abs(open_qty))
                
                # 按比例分摊手续费
                open_comm_fraction = match_qty / abs(open_qty)
                open_comm_allocated = open_pos['comm'] * open_comm_fraction
                
                close_comm_fraction = match_qty / abs(qty)
                close_comm_allocated = comm * close_comm_fraction
                
                # 计算已实现盈亏 (Realized PnL)
                # 做多平仓: (平仓价 - 开仓价) * 数量 * 乘数
                # 做空平仓: (开仓价 - 平仓价) * 数量 * 乘数  (即等价于 (平仓 - 开仓) * open_sign)
                realized_pnl = (price - open_pos['price']) * open_sign * match_qty * mult
                total_comm = open_comm_allocated + close_comm_allocated
                net_pnl = realized_pnl + total_comm
                
                # 记录这笔已平仓交易
                closed_trades.append({
                    '标的代码': sym,
                    '开仓日期': open_pos['date'],
                    '平仓日期': date,
                    '方向': '做多 (Long)' if open_sign > 0 else '做空 (Short)',
                    '匹配数量': match_qty,
                    '开仓均价': open_pos['price'],
                    '平仓价': price,
                    '乘数': mult,
                    '分摊佣金': total_comm,
                    '净盈亏': net_pnl
                })
                
                # 扣减剩余数量和佣金
                qty -= match_qty * trade_sign
                comm -= close_comm_allocated
                
                open_pos['qty'] -= match_qty * open_sign
                open_pos['comm'] -= open_comm_allocated
                
                # 如果该笔开仓单已被完全平掉，则移出队列
                if abs(open_pos['qty']) < 1e-6:
                    open_positions[sym].pop(0)
                    
    return pd.DataFrame(closed_trades)

def plot_profit_calendar(filtered_df: pd.DataFrame, start_date, end_date):
    """绘制类似 GitHub 贡献图的盈亏日历热力图"""
    if filtered_df.empty:
        return go.Figure()

    # 1. 聚合每日数据
    daily_stats = filtered_df.groupby(filtered_df['平仓日期'].dt.date).agg(
        Net_PnL=('净盈亏', 'sum'),
        Trades=('净盈亏', 'count')
    ).reset_index()
    daily_stats.rename(columns={'平仓日期': 'Date'}, inplace=True)

    # 2. 生成完整日期范围
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df_cal = pd.DataFrame({'Date': all_dates.date})
    df_cal = df_cal.merge(daily_stats, on='Date', how='left').fillna({'Net_PnL': 0, 'Trades': 0})

    # 3. 计算用于绘图的坐标 (WeekIdx作为X轴，Weekday作为Y轴)
    start_dt = pd.to_datetime(start_date)
    df_cal['Weekday'] = pd.to_datetime(df_cal['Date']).dt.weekday  # 0=Mon, 6=Sun
    df_cal['DayOffset'] = (pd.to_datetime(df_cal['Date']) - start_dt).dt.days
    df_cal['WeekIdx'] = (df_cal['DayOffset'] + start_dt.weekday()) // 7

    weeks = df_cal['WeekIdx'].unique()
    week_labels = [f"第{int(w)+1}周" for w in weeks]
    y_labels = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日']

    # 4. 构建数据矩阵
    z_data = np.zeros((7, len(weeks)))
    hover_data = np.empty((7, len(weeks), 2), dtype=object)
    
    for _, row in df_cal.iterrows():
        w = int(row['WeekIdx'])
        d = int(row['Weekday'])
        z_data[d, w] = row['Net_PnL']
        hover_data[d, w, 0] = str(row['Date'])
        hover_data[d, w, 1] = int(row['Trades'])

    # 5. 绘制 Plotly 热力图
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=week_labels,
        y=y_labels,
        customdata=hover_data,
        colorscale='RdYlGn',  # 红(亏)-黄(平)-绿(盈)
        zmid=0,               # 确保 0 对应颜色中心（黄色/浅色）
        xgap=4, ygap=4,       # 方块间隙
        hovertemplate=(
            "<b>日期:</b> %{customdata[0]}<br>"
            "<b>交易笔数:</b> %{customdata[1]} 笔<br>"
            "<b>当日净盈亏:</b> $%{z:.2f}<extra></extra>"
        )
    ))

    fig.update_layout(
        title="📅 每日平仓盈亏日历",
        xaxis_visible=False,
        yaxis=dict(autorange="reversed"), # 反转Y轴让星期一显示在最上方
        plot_bgcolor='white',
        height=350,
        margin=dict(t=50, b=20, l=50, r=20)
    )
    return fig

# ==========================================
# UI 渲染部分
# ==========================================
st.title("📈 交易表现分析与盈亏日历")
st.markdown("通过上传券商导出的交易流水（目前支持 IBKR 格式），自动匹配 FIFO 开平仓，并生成交易表现和日历统计。")

# 1. 上传文件区 (支持拖拽)
uploaded_file = st.file_uploader("📂 拖拽或点击上传 CSV 文件", type=['csv'])

if uploaded_file is not None:
    # 2. 读取与清洗数据
    file_content = uploaded_file.getvalue().decode('utf-8')
    df_raw = parse_and_clean_data(file_content)
    
    if df_raw.empty:
        st.error("⚠️ 未能从文件中解析出有效的交易数据，请确保上传的是格式正确的 CSV 报表。")
    else:
        # 3. 匹配交易
        trades_df = match_trades_fifo(df_raw)
        
        if trades_df.empty:
            st.warning("⚠️ 未能匹配到任何已完成的完整交易（可能是全开仓或全平仓）。")
        else:
            # 4. 时间筛选器
            min_date = trades_df['平仓日期'].min().date()
            max_date = trades_df['平仓日期'].max().date()
            
            st.divider()
            st.subheader("🗓️ 统计时间范围选择")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input("开始日期 (基于平仓日)", min_date, min_value=min_date, max_value=max_date)
            with col_d2:
                end_date = st.date_input("结束日期 (基于平仓日)", max_date, min_value=min_date, max_value=max_date)
            
            # 过滤数据
            mask = (trades_df['平仓日期'].dt.date >= start_date) & (trades_df['平仓日期'].dt.date <= end_date)
            filtered_df = trades_df.loc[mask].copy()
            
            if filtered_df.empty:
                st.info("选定的时间范围内没有已平仓交易。")
            else:
                # 5. 计算统计指标
                total_trades = len(filtered_df)
                gross_profit = filtered_df[filtered_df['净盈亏'] > 0]['净盈亏'].sum()
                gross_loss = filtered_df[filtered_df['净盈亏'] <= 0]['净盈亏'].sum()
                net_pnl = gross_profit + gross_loss
                
                win_trades = len(filtered_df[filtered_df['净盈亏'] > 0])
                loss_trades = len(filtered_df[filtered_df['净盈亏'] <= 0])
                win_rate = win_trades / total_trades if total_trades > 0 else 0
                
                # 盈利因子: 绝对值 (总盈利 / 总亏损)
                profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
                
                # 平均盈亏
                avg_win = gross_profit / win_trades if win_trades > 0 else 0
                avg_loss = abs(gross_loss / loss_trades) if loss_trades > 0 else 0
                avg_rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
                
                total_commission = filtered_df['分摊佣金'].sum()

                # 6. 展示核心指标
                st.subheader("📊 交易绩效概览")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("💰 净盈亏 (Net PnL)", f"${net_pnl:,.2f}")
                m2.metric("🎯 胜率 (Win Rate)", f"{win_rate * 100:.1f}%")
                m3.metric("⚖️ 盈利因子 (Profit Factor)", f"{profit_factor:.2f}")
                m4.metric("📊 盈亏比 (Avg RR)", f"{avg_rr:.2f}")
                
                m5, m6, m7, m8 = st.columns(4)
                m5.metric("📈 盈利 / 📉 亏损 笔数", f"{win_trades} / {loss_trades} 笔")
                m6.metric("🟢 总盈利 (Gross Profit)", f"${gross_profit:,.2f}")
                m7.metric("🔴 总亏损 (Gross Loss)", f"${gross_loss:,.2f}")
                m8.metric("💸 总手续费 (Commission)", f"${total_commission:,.2f}")

                # 7. 展示日历热力图
                st.divider()
                fig_cal = plot_profit_calendar(filtered_df, start_date, end_date)
                st.plotly_chart(fig_cal, use_container_width=True)

                # 8. 展示交易明细表
                st.subheader("📝 已平仓交易明细 (Closed Trades)")
                # 格式化金额列以便于阅读
                display_df = filtered_df.copy()
                display_df['净盈亏'] = display_df['净盈亏'].apply(lambda x: f"${x:,.2f}")
                display_df['分摊佣金'] = display_df['分摊佣金'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(display_df.sort_values('平仓日期', ascending=False), use_container_width=True)