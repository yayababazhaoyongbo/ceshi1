import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 1. 配置中心 =================
DB_NAME = "market_data_ultimate.db"
THREAD_COUNT = 15
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# ================= 2. 核心过滤与数据获取 =================
def is_valid_stock(code, name):
    if code.startswith(('3', '8', '4')): return False
    blacklist = ["ST", "退", "指数", "ETF", "LOF", "基金", "债", "转", "平"]
    if any(x in name for x in blacklist): return False
    return True

def get_data(code):
    clean_code = str(code).zfill(6)
    # 如果是上证指数作为基准
    if clean_code == '000001':
        symbol = 'sh000001'
    else:
        if not clean_code.startswith(('60', '00', '001', '002')): return None, None
        symbol = ('sh' if clean_code.startswith('6') else 'sz') + clean_code
        
    url = f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get?param={symbol},day,,,250,qfq"
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": UA}, verify=False)
        data = resp.json()['data'][symbol]
        name = data.get('qt', {}).get(symbol, ["", "未知"])[1]
        
        if clean_code != '000001' and not is_valid_stock(clean_code, name): return None, None
        
        data_list = data.get('qfqday', data.get('day', []))
        if len(data_list) < 60: return None, None
        
        df = pd.DataFrame(data_list).iloc[:, [0, 2, 5]]
        df.columns = ['日期', '收盘', '成交量']
        df[['收盘', '成交量']] = df[['收盘', '成交量']].apply(pd.to_numeric)
        return df, name
    except: return None, None

# ================= 3. 灵魂均线核心算法 (复用原有逻辑) =================
def find_best_ma(df):
    """
    动态计算最优均线（灵魂均线）。
    由于简化演示，这里用一个相对快速的遍历逻辑替代极其复杂的全面回测。
    核心思想：寻找在过去一段时间内，作为支撑线表现最好的 EMA 周期。
    """
    best_ma = 0
    best_score = -9999
    c = df['收盘']
    
    # 遍历常用均线周期
    for period in [10, 20, 30, 60, 120]:
        ema = c.ewm(span=period, adjust=False).mean()
        # 简单评分逻辑：处于均线上方的天数比例 - 跌破的最大幅度惩罚
        above_days = (c > ema).sum()
        max_drawdown = ((ema - c) / ema).clip(lower=0).max()
        score = above_days - (max_drawdown * 1000) # 惩罚深跌
        
        if score > best_score:
            best_score = score
            best_ma = period
            
    return best_ma, best_score

# ================= 4. 高级量化指标计算 =================
def calculate_volatility(df, period=60):
    """计算年化波动率"""
    returns = df['收盘'].pct_change().dropna()
    # 乘以 sqrt(252) 年化
    volatility = returns.tail(period).std() * np.sqrt(252)
    return volatility

def calculate_momentum(stock_df, index_df, period=20):
    """计算相对大盘的截面动量 (超额收益)"""
    # 确保时间对齐，这里简易处理，直接取最后 period 天的涨跌幅
    stock_return = (stock_df['收盘'].iloc[-1] - stock_df['收盘'].iloc[-period-1]) / stock_df['收盘'].iloc[-period-1]
    index_return = (index_df['收盘'].iloc[-1] - index_df['收盘'].iloc[-period-1]) / index_df['收盘'].iloc[-period-1]
    return stock_return - index_return

# ================= 5. 基建与扫描逻辑 =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    # 增加了 best_ma 字段用于存储灵魂均线周期
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_genes (code TEXT PRIMARY KEY, name TEXT, best_ma INTEGER)''')
    conn.commit(); conn.close()

init_db()
st.set_page_config(layout="wide")
st.title("🚀 量化狙击系统: 灵魂均线 + 动量共振")

tabs = st.tabs(["🏗️ 基因基建", "🎯 基础放量突破", "⚡ 三维立体狙击 (低波+动量)"])

with tabs[0]:
    st.markdown("### 步骤1：全市场基建与灵魂均线计算")
    if st.button("开始全市场基建"):
        pool = [f"{p}{i:03d}" for p in ['600','601','603','605','000','001','002'] for i in range(1000)]
        conn = sqlite3.connect(DB_NAME)
        existing = pd.read_sql("SELECT code FROM stock_genes", conn)['code'].astype(str).tolist()
        conn.close()
        
        todo = [c for c in pool if c not in existing]
        bar = st.progress(0.0)
        
        def infra_task(c):
            df, name = get_data(c)
            if df is not None:
                # 基建时直接算出它的灵魂均线
                best_ma_period, _ = find_best_ma(df)
                return (c, name, best_ma_period)
            return None

        results_to_insert = []
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
            futs = {ex.submit(infra_task, c): c for c in todo}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: results_to_insert.append(res)
                if len(todo) > 0 and i % 50 == 0: bar.progress((i+1)/len(todo))
        
        if results_to_insert:
            conn = sqlite3.connect(DB_NAME)
            conn.executemany("INSERT OR REPLACE INTO stock_genes VALUES (?,?,?)", results_to_insert)
            conn.commit(); conn.close()
        bar.progress(1.0)
        st.success("基建与灵魂均线计算完成！")

with tabs[1]:
    st.markdown("### 步骤2：常规扫描 (复用旧逻辑)")
    # 这里保留你原来的普通突破逻辑，略...
    st.info("请使用 Tab 3 的三维狙击。")

with tabs[2]:
    st.markdown("### 步骤3：三维漏斗过滤 (核心)")
    st.markdown("寻找 **低波动蓄势** 且 **异动跑赢大盘** 且 **突破自身灵魂均线** 的猎物。")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_vol = st.slider("最大年化波动率阈值 (过滤振幅过大)", 0.1, 1.0, 0.4, 0.05)
    with col2:
        min_mom = st.slider("最低相对大盘动量 (%) (必须跑赢大盘)", -5.0, 20.0, 5.0, 1.0)
    with col3:
        vol_m = st.slider("突破时成交量放大倍数", 1.0, 3.0, 1.5)

    if st.button("⚡ 执行三维共振扫描"):
        conn = sqlite3.connect(DB_NAME)
        genes = pd.read_sql("SELECT * FROM stock_genes", conn)
        conn.close()
        
        scan_bar = st.progress(0.0)
        status_text = st.empty()
        status_text.text("正在获取大盘基准数据...")
        
        # 1. 获取大盘基准
        index_df, _ = get_data('000001')
        if index_df is None:
            st.error("大盘数据获取失败，无法计算动量。")
            st.stop()
            
        hits = []
        status_text.text("正在执行三维漏斗扫描...")
        
        def ultimate_scan_task(row):
            df, name = get_data(row['code'])
            if df is not None:
                c, v = df['收盘'], df['成交量']
                
                # --- 漏斗第一级：低波动过滤 ---
                volatility = calculate_volatility(df)
                if volatility > max_vol: return None
                
                # --- 漏斗第二级：动量共振 (跑赢大盘) ---
                momentum = calculate_momentum(df, index_df)
                if momentum * 100 < min_mom: return None
                
                # --- 漏斗第三级：灵魂均线突破 ---
                # 读取基建时算好的本命均线周期
                best_ma_period = row['best_ma'] if pd.notna(row['best_ma']) else 60
                ema_soul = c.ewm(span=best_ma_period, adjust=False).mean()
                
                # 严格金叉放量
                if c.iloc[-1] > ema_soul.iloc[-1] and c.iloc[-2] <= ema_soul.iloc[-2]:
                    v_avg = v.shift(1).rolling(20).mean()
                    if v.iloc[-1] > v_avg.iloc[-1] * vol_m:
                        return {
                            "代码": row['code'],
                            "名称": row['name'],
                            "本命均线": f"EMA{int(best_ma_period)}",
                            "当前价": round(c.iloc[-1], 2),
                            "波动率 (低波)": f"{volatility:.2f}",
                            "超额动量 (强势)": f"{momentum*100:.2f}%",
                            "今日量比": round(v.iloc[-1] / v_avg.iloc[-1], 2)
                        }
            return None

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
            futs = [ex.submit(ultimate_scan_task, r) for _, r in genes.iterrows()]
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: hits.append(res)
                if len(genes) > 0 and i % 50 == 0: 
                    scan_bar.progress((i+1)/len(genes))
                    
        scan_bar.progress(1.0)
        status_text.text("扫描完毕。")
        
        if hits:
            st.balloons()
            st.dataframe(pd.DataFrame(hits))
        else:
            st.warning("条件极为严苛，当前市场暂无完全符合【三维共振】的标的。建议适当放宽动量或波动率条件。")