import streamlit as st
import pandas as pd
import numpy as np
import requests
import sqlite3
import time
import random
import threading
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 1. 配置中心 =================
DB_NAME = "market_data_ultimate.db"
THREAD_COUNT = 20  

# 引入真实浏览器 UA 池，实现“幽灵”伪装，彻底防封禁
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0"
]

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(pool_connections=THREAD_COUNT, pool_maxsize=THREAD_COUNT, max_retries=retries)
        thread_local.session.mount('http://', adapter)
        thread_local.session.mount('https://', adapter)
    return thread_local.session

# ================= 2. 核心过滤与数据获取 =================
def is_valid_stock(code, name):
    if code.startswith(('3', '8', '4')): return False
    blacklist = ["ST", "退", "指数", "ETF", "LOF", "基金", "债", "转", "平"]
    if any(x in name for x in blacklist): return False
    return True

def get_data(code):
    clean_code = str(code).zfill(6)
    if clean_code == '000001':
        symbol = 'sh000001'
    else:
        if not clean_code.startswith(('60', '00', '001', '002')): return None, None
        symbol = ('sh' if clean_code.startswith('6') else 'sz') + clean_code
        
    url = f"https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get?param={symbol},day,,,500,qfq"
    try:
        time.sleep(random.uniform(0.01, 0.05))
        session = get_session()
        headers = {"User-Agent": random.choice(UA_LIST)}
        resp = session.get(url, timeout=5, headers=headers, verify=False)
        data = resp.json()['data'][symbol]
        name = data.get('qt', {}).get(symbol, ["", "未知"])[1]
        
        if clean_code != '000001' and not is_valid_stock(clean_code, name): return None, None
        
        data_list = data.get('qfqday', data.get('day', []))
        if len(data_list) < 60: return None, None
        
        df = pd.DataFrame(data_list).iloc[:, [0, 2, 5]]
        df.columns = ['日期', '收盘', '成交量']
        df[['收盘', '成交量']] = df[['收盘', '成交量']].apply(pd.to_numeric)
        
        df['最低'] = df['收盘'] * 0.985 
        return df, name
    except: return None, None

# ================= 3. 灵魂均线核心算法 =================
def find_best_ma(df):
    best_ma = 0
    best_score = -9999
    c = df['收盘']
    for period in [10, 20, 30, 60, 120]:
        ema = c.ewm(span=period, adjust=False).mean()
        above_days = (c > ema).sum()
        max_drawdown = ((ema - c) / ema).clip(lower=0).max()
        score = above_days - (max_drawdown * 1000)
        if score > best_score:
            best_score = score
            best_ma = period
    return best_ma, best_score

# ================= 4. 高级量化指标计算 =================
def calculate_volatility(df, period=60):
    returns = df['收盘'].pct_change().dropna()
    volatility = returns.tail(period).std() * np.sqrt(252)
    return volatility

def calculate_momentum(stock_df, index_df, period=20):
    try:
        stock_return = (stock_df['收盘'].iloc[-1] - stock_df['收盘'].iloc[-period-1]) / stock_df['收盘'].iloc[-period-1]
        index_return = (index_df['收盘'].iloc[-1] - index_df['收盘'].iloc[-period-1]) / index_df['收盘'].iloc[-period-1]
        return stock_return - index_return
    except: return 0

def calculate_advanced_indicators(df, dynamic_params=None):
    c = df['收盘']
    v = df['成交量']
    if len(c) < 260: return False, None, None, None, False, False
    
    # 继承上一版的严苛特征
    target_sqz = 0.15   
    target_roc = 2.5    
    target_vol = 1.7    
    # 【新增防飞刀参数】：是否启用斜率保护
    reject_downtrend = True
    
    if dynamic_params:
        target_sqz = dynamic_params.get("max_sqz", target_sqz)
        target_roc = dynamic_params.get("min_roc", target_roc)
        target_vol = dynamic_params.get("min_vol", target_vol)
        reject_downtrend = dynamic_params.get("reject_downtrend", True)
    
    ma20 = c.rolling(20).mean()
    
    # 【重磅修复：趋势斜率锁】
    # 计算20日均线过去3天的倾斜度。如果还在快速杀跌，说明是下跌中继，直接一票否决！
    ma20_slope = (ma20.iloc[-1] - ma20.iloc[-4]) / ma20.iloc[-4]
    is_falling_knife = (ma20_slope < -0.015) # 3天内20日线向下俯冲超过1.5%视为飞刀
    
    std20 = c.rolling(20).std()
    bbw = (4 * std20) / ma20
    bbw_min_120 = bbw.rolling(120).min()
    bbw_max_120 = bbw.rolling(120).max()
    
    squeeze_score = ((bbw.iloc[-1] - bbw_min_120.iloc[-1]) / (bbw_max_120.iloc[-1] - bbw_min_120.iloc[-1] + 1e-9)) * 100
    squeeze_on = (squeeze_score <= target_sqz)
    
    v_ma3 = v.rolling(3).mean()
    v_ma20 = v.rolling(20).mean()
    volume_dry_up = v_ma3.iloc[-2] < (v_ma20.iloc[-2] * 0.70)
    vol_ratio = v.iloc[-1] / v_ma20.iloc[-1]
    vol_pulse = (vol_ratio >= target_vol)
    
    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    momentum_pulse = hist.iloc[-1] > hist.iloc[-2]
    
    macd_cross = (macd.iloc[-2] < signal.iloc[-2]) and (macd.iloc[-1] > signal.iloc[-1])
    roc12 = (c.diff(12) / c.shift(12)) * 100
    roc_pulse = roc12.iloc[-1] > roc12.iloc[-2]
    roc_cross = (roc12.iloc[-2] < 0) and (roc12.iloc[-1] > 0)
    
    is_double_cross = macd_cross and roc_cross
    
    if roc12.iloc[-1] < target_roc:
        roc_pulse = False
        
    price_stable = c.iloc[-1] > c.iloc[-4] * 0.96
    
    ma250 = c.rolling(250).mean()
    annual_support = False
    if pd.notna(ma250.iloc[-1]):
        recent_low = c.tail(5).min() * 0.985 
        if (recent_low >= ma250.iloc[-1] * 0.98) and (recent_low <= ma250.iloc[-1] * 1.05) and (c.iloc[-1] > ma250.iloc[-1]):
            annual_support = True
            
    # 【增加飞刀过滤】：如果开启了规避主跌浪，且正在自由落体，则强制否决
    if reject_downtrend and is_falling_knife:
        is_spring_loaded = False
    else:
        is_spring_loaded = squeeze_on and momentum_pulse and roc_pulse and price_stable and (volume_dry_up or vol_pulse)
    
    return is_spring_loaded, roc12.iloc[-1], squeeze_score, vol_ratio, annual_support, is_double_cross

# ================= 5. 基建与 UI =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_genes (code TEXT PRIMARY KEY, name TEXT, best_ma INTEGER)''')
    conn.commit(); conn.close()

init_db()
st.set_page_config(layout="wide")
st.title("🚀 量化狙击系统: 趋势斜率护城河版")

tabs = st.tabs(["🏗️ 基因基建", "⚡ 实盘扫描", "⏳ 时光机 (全量回测)", "🏆 归因分析提炼"])

# ----------------- Tab 1: 基因基建 -----------------
with tabs[0]:
    st.markdown("### 全市场基建与灵魂均线计算")
    if st.button("开始全市场基建"):
        pool = [f"{p}{i:03d}" for p in ['600','601','603','605','000','001','002'] for i in range(1000)]
        conn = sqlite3.connect(DB_NAME)
        existing = pd.read_sql("SELECT code FROM stock_genes", conn)['code'].astype(str).tolist()
        conn.close()
        todo = [c for c in pool if c not in existing]
        bar = st.progress(0.0)
        status_text = st.empty()
        
        def infra_task(c):
            df, name = get_data(c)
            if df is not None:
                best_ma_period, _ = find_best_ma(df)
                return (c, name, best_ma_period)
            return None

        results_to_insert = []
        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
            futs = {ex.submit(infra_task, c): c for c in todo}
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: results_to_insert.append(res)
                if len(todo) > 0 and i % 10 == 0: 
                    bar.progress((i+1)/len(todo))
                    status_text.text(f"基建进度: {i+1} / {len(todo)} | 已完成计算: {len(results_to_insert)} 只")
        
        if results_to_insert:
            conn = sqlite3.connect(DB_NAME)
            conn.executemany("INSERT OR REPLACE INTO stock_genes VALUES (?,?,?)", results_to_insert)
            conn.commit(); conn.close()
        bar.progress(1.0)
        st.success("基建与灵魂均线计算完成！")

# ----------------- Tab 2: 实盘扫描 -----------------
with tabs[1]:
    st.markdown("### 今日实时策略扫描")
    strategy_choice = st.radio("选择狙击策略:", ["三维共振 (右侧均线突破)", "前瞻预判 (左侧形态压缩)"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_vol = st.slider("最大年化波动率阈值", 0.1, 1.0, 0.4, 0.05)
    with col2:
        min_mom = st.slider("最低相对大盘动量 (%)", -5.0, 20.0, 5.0, 1.0)
    with col3:
        vol_m = st.slider("突破时成交量放大倍数", 1.0, 3.0, 1.5)
        
    st.markdown("---")
    # 【UI新增】：给用户规避主跌浪的选择权
    reject_knife = st.checkbox("🛡️ 规避下降通道 (要求近期20日均线斜率不得向下俯冲)", value=True)

    if st.button("⚡ 执行今日全市场扫描"):
        conn = sqlite3.connect(DB_NAME)
        genes = pd.read_sql("SELECT * FROM stock_genes", conn)
        conn.close()
        
        scan_bar = st.progress(0.0)
        status_text = st.empty()
        index_df, _ = get_data('000001')
        hits = []
        
        def real_scan_task(row):
            df, name = get_data(row['code'])
            if df is None or len(df) < 260: return None
            
            c, v = df['收盘'], df['成交量']
            if "三维共振" in strategy_choice:
                volatility = calculate_volatility(df)
                if volatility > max_vol: return None
                momentum = calculate_momentum(df, index_df) if index_df is not None else 0
                if momentum * 100 < min_mom: return None
                
                best_ma_period = row['best_ma'] if pd.notna(row['best_ma']) else 60
                ema_soul = c.ewm(span=best_ma_period, adjust=False).mean()
                
                if c.iloc[-1] > ema_soul.iloc[-1] and c.iloc[-2] <= ema_soul.iloc[-2]:
                    v_avg = v.shift(1).rolling(20).mean()
                    if v.iloc[-1] > v_avg.iloc[-1] * vol_m:
                        return {"代码": row['code'], "名称": row['name'], "策略": "三维共振", "当前价": round(c.iloc[-1], 2)}
            else:
                dyn_params = {"reject_downtrend": reject_knife}
                is_spring, roc_val, sqz_score, vol_ratio, is_annual_bounce, is_double_cross = calculate_advanced_indicators(df, dyn_params)
                if is_spring:
                    tag = "🔴极致压缩"
                    if is_annual_bounce: tag += " + 🐉年线回踩"
                    if is_double_cross: tag += " + 🌟双金叉"
                    return {
                        "代码": row['code'], "名称": row['name'], "形态标签": tag, 
                        "当前价": round(c.iloc[-1], 2), "ROC(12)": f"{roc_val:.2f}%", "压缩评分": f"{sqz_score:.1f}分"
                    }
            return None

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
            futs = [ex.submit(real_scan_task, r) for _, r in genes.iterrows()]
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: hits.append(res)
                if len(genes) > 0 and i % 10 == 0: 
                    scan_bar.progress((i+1)/len(genes))
                    status_text.text(f"正在扫描... 进度: {i+1} / {len(genes)} | 🎯 已发现目标: {len(hits)} 个")
                    
        scan_bar.progress(1.0)
        status_text.text(f"扫描完毕。共发现 {len(hits)} 个目标。")
        if hits: st.dataframe(pd.DataFrame(hits))
        else: st.warning("当前市场暂无完全符合该策略的标的。")

# ----------------- Tab 3 & Tab 4 维持上一版状态 -----------------
# ... (为精简上下文，后续Tab保持原逻辑即可，它们也会自动继承这个斜率防坠落判定)
with tabs[2]:
    st.markdown("### ⏳ 时光机：历史扫描与全量特征捕捉")
    # ... 省略重复内容，功能不变
    
with tabs[3]:
    st.markdown("### 🏆 数据归因与提炼 (专为喂给大模型设计)")
    # ... 省略重复内容，功能不变
