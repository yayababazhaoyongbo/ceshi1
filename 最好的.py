import streamlit as st
st.set_page_config(layout="wide", page_title="终极量化狙击系统", page_icon="🎯")

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
THREAD_COUNT = 10  

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0"
]

@st.cache_resource
def get_global_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=THREAD_COUNT, pool_maxsize=THREAD_COUNT, max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

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
        session = get_global_session()
        headers = {"User-Agent": random.choice(UA_LIST)}
        resp = session.get(url, timeout=6, headers=headers, verify=False)
        
        if resp.status_code != 200:
            return None, None
            
        data = resp.json()['data'][symbol]
        name = data.get('qt', {}).get(symbol, ["", "未知"])[1]
        
        if clean_code != '000001' and not is_valid_stock(clean_code, name): return None, None
        
        data_list = data.get('qfqday', data.get('day', []))
        if len(data_list) < 60: return None, None
        
        df = pd.DataFrame(data_list).iloc[:, [0, 1, 2, 3, 4, 5]]
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        df[['开盘', '收盘', '最高', '最低', '成交量']] = df[['开盘', '收盘', '最高', '最低', '成交量']].apply(pd.to_numeric)
        
        return df, name
    except Exception: 
        return None, None

# ================= 3. 通道震荡与双通道嵌套算法 (引入严苛过滤) =================
def detect_regression_channel(df, max_period=90, direction='下降通道', require_bottom=True, min_width=30, min_touches=3, strict_close=True):
    """
    【升级版】严苛通道猎手
    增加严格的空间限制 (min_width)、触碰次数限制 (min_touches)、以及收盘价收敛控制 (strict_close)。
    """
    actual_max = min(max_period, len(df))
    if actual_max < 30: return None
    
    # 动态寻优循环：不断缩小窗口，寻找最完美契合的那一条通道
    for p in range(actual_max, 29, -5):
        df_slice = df.tail(p)
        x = np.arange(p)
        y = df_slice['收盘'].values
        
        y_norm = y / y[0]
        slope_norm, _ = np.polyfit(x, y_norm, 1)
        
        if direction == '下降通道' and slope_norm > -0.001: continue
        if direction == '上升通道' and slope_norm < 0.001: continue
        
        slope, intercept = np.polyfit(x, y, 1)
        reg_line = slope * x + intercept
        
        # 使用 1.6 倍标准差作为通道包络线，保证数学意义上包裹住绝大部分震荡区间
        std_dev = np.std(y - reg_line)
        upper_band = reg_line + 1.6 * std_dev
        lower_band = reg_line - 1.6 * std_dev
        
        # 【严苛条件 3】：主力底线控制 - 收盘价严格在通道内 (容许 1% 的数学误差缓冲)
        if strict_close:
            closes_outside = np.sum((y > upper_band * 1.01) | (y < lower_band * 0.99))
            if closes_outside > 0:
                continue  # 只要有任何一天收盘价没收回通道内，直接淘汰该周期
        
        highs = df_slice['最高'].values
        lows = df_slice['最低'].values
        
        # 统计盘中高低点触碰甚至刺穿边界的次数 (允许刺穿，即 high >= upper_band)
        upper_touches = np.sum(highs >= upper_band * 0.985)
        lower_touches = np.sum(lows <= lower_band * 1.015)
        
        # 【严苛条件 2】：顶部或底部触碰至少 min_touches 次 (且必须是完整通道，弱势边至少2次)
        if max(upper_touches, lower_touches) < min_touches: continue
        if min(upper_touches, lower_touches) < 2: continue
            
        channel_width = (upper_band[-1] - lower_band[-1]) / lower_band[-1] * 100
        
        # 【严苛条件 1】：通道空间(下跌或上涨幅度)不少于 min_width %
        if channel_width < min_width: continue
            
        current_close = y[-1]
        dist_to_lower = (current_close - lower_band[-1]) / lower_band[-1] * 100
        
        if require_bottom and (dist_to_lower > 3.0 or dist_to_lower < -2.0): continue
            
        # 只要找到了，就直接返回
        return {
            "通道方向": "↘️ 下降震荡" if slope_norm < 0 else "↗️ 上升震荡",
            "实际天数": f"{p}天",
            "通道空间": f"{channel_width:.1f}%",
            "上轨触碰": f"{upper_touches}次",
            "下轨触碰": f"{lower_touches}次",
            "当前价": round(current_close, 2),
            "距下轨(买点)": f"{dist_to_lower:.2f}%"
        }
    return None

def detect_dual_channel(df, max_macro_p=180, max_micro_p=100, direction='下降通道'):
    actual_mac = min(max_macro_p, len(df))
    actual_mic = min(max_micro_p, len(df))
    if actual_mac < 60: return None
    
    valid_macros = []
    for mac_p in range(actual_mac, 59, -10):
        df_mac = df.tail(mac_p)
        x_mac = np.arange(mac_p)
        y_mac = df_mac['收盘'].values
        slope_mac, int_mac = np.polyfit(x_mac, y_mac, 1)
        
        slope_mac_norm = slope_mac / y_mac[0]
        if direction == '下降通道' and slope_mac_norm > -0.001: continue
        if direction == '上升通道' and slope_mac_norm < 0.001: continue
        
        reg_mac = slope_mac * x_mac + int_mac
        std_mac = np.std(y_mac - reg_mac)
        lower_mac = reg_mac - 1.5 * std_mac 
        
        current_close = y_mac[-1]
        dist_to_mac_lower = (current_close - lower_mac[-1]) / lower_mac[-1] * 100
        
        if -3.0 <= dist_to_mac_lower <= 4.0:
            valid_macros.append((mac_p, lower_mac, slope_mac_norm, dist_to_mac_lower))
            break 
            
    if not valid_macros:
        return None
        
    mac_p, lower_mac, slope_mac_norm, dist_to_mac_lower = valid_macros[0]
    
    for mic_p in range(actual_mic, 19, -5):
        if mic_p >= mac_p: continue 
        
        df_mic = df.tail(mic_p)
        x_mic = np.arange(mic_p)
        y_mic = df_mic['收盘'].values
        slope_mic, int_mic = np.polyfit(x_mic, y_mic, 1)
        
        reg_mic = slope_mic * x_mic + int_mic
        std_mic = np.std(y_mic - reg_mic)
        lower_mic = reg_mic - 1.5 * std_mic 
        
        dist_to_mic_lower = (current_close - lower_mic[-1]) / lower_mic[-1] * 100
        
        if dist_to_mic_lower < 1.0:
            return {
                "形态": "🪆双通道诱空(破底翻)",
                "大通道方向": "↘️ 下降" if slope_mac_norm < 0 else "↗️ 上升",
                "大通道(实)": f"{mac_p}天",
                "小通道(实)": f"{mic_p}天",
                "当前价": round(current_close, 2),
                "距大下轨(支撑)": f"{dist_to_mac_lower:.2f}%",
                "距小下轨(破位)": f"{dist_to_mic_lower:.2f}% (诱空)"
            }
    return None

# ================= 4. 旧版指标计算保留 =================
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
    
    target_sqz = 0.15   
    target_roc = 2.5    
    target_vol = 1.7    
    reject_downtrend = True
    
    if dynamic_params:
        target_sqz = dynamic_params.get("max_sqz", target_sqz)
        target_roc = dynamic_params.get("min_roc", target_roc)
        target_vol = dynamic_params.get("min_vol", target_vol)
        reject_downtrend = dynamic_params.get("reject_downtrend", True)
    
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    
    ma20_slope_10d = (ma20.iloc[-1] - ma20.iloc[-10]) / ma20.iloc[-10] if pd.notna(ma20.iloc[-10]) else 0
    is_falling_knife = (ma20_slope_10d < -0.02)
    
    ma60_slope_20d = (ma60.iloc[-1] - ma60.iloc[-20]) / ma60.iloc[-20] if pd.notna(ma60.iloc[-20]) else 0
    price_suppressed_by_60 = c.iloc[-1] < ma60.iloc[-1]
    is_macro_downtrend = (ma60_slope_20d < -0.01) and price_suppressed_by_60
    
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
            
    if reject_downtrend and (is_falling_knife or is_macro_downtrend):
        is_spring_loaded = False
    else:
        is_spring_loaded = squeeze_on and momentum_pulse and roc_pulse and price_stable and (volume_dry_up or vol_pulse)
    
    return is_spring_loaded, roc12.iloc[-1], squeeze_score, vol_ratio, annual_support, is_double_cross

# ================= 5. 基建与 UI =================
@st.cache_resource
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME, timeout=15, check_same_thread=False)
        conn.execute('''CREATE TABLE IF NOT EXISTS stock_genes (code TEXT PRIMARY KEY, name TEXT, best_ma INTEGER)''')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

db_ready = init_db()

st.title("🚀 量化狙击系统: 终极地狱漏斗版")

if not db_ready:
    st.error("云端存储环境初始化失败，请稍后刷新重试。")
    st.stop()

tabs = st.tabs(["🏗️ 基因基建", "⚡ 实盘扫描", "⏳ 时光机回测", "🏆 归因提炼", "📈 规律通道猎手"])

# ----------------- Tab 1: 基因基建 -----------------
with tabs[0]:
    st.markdown("### 全市场基建与灵魂均线计算")
    if st.button("开始全市场基建"):
        pool = [f"{p}{i:03d}" for p in ['600','601','603','605','000','001','002'] for i in range(1000)]
        try:
            conn = sqlite3.connect(DB_NAME, timeout=15)
            existing = pd.read_sql("SELECT code FROM stock_genes", conn)['code'].astype(str).tolist()
            conn.close()
        except: existing = []
            
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
            conn = sqlite3.connect(DB_NAME, timeout=15)
            conn.executemany("INSERT OR REPLACE INTO stock_genes VALUES (?,?,?)", results_to_insert)
            conn.commit(); conn.close()
        bar.progress(1.0)
        st.success("基建与灵魂均线计算完成！")

# ----------------- Tab 2: 实盘扫描 -----------------
with tabs[1]:
    st.markdown("### 今日实时策略扫描 (主升浪探测器)")
    strategy_choice = st.radio("选择狙击策略:", [
        "三维共振 (右侧均线突破)", 
        "前瞻预判 (左侧形态压缩)",
        "💥 跳空爆发 (缺口+大阳线)"
    ], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    with col1: max_vol = st.slider("最大年化波动率阈值", 0.1, 1.0, 0.4, 0.05)
    with col2: min_mom = st.slider("最低相对大盘动量 (%)", -5.0, 20.0, 5.0, 1.0)
    with col3: vol_m = st.slider("突破时成交量放大倍数", 1.0, 3.0, 1.5)
        
    st.markdown("---")
    reject_knife = st.checkbox("🛡️ 严禁接飞刀：自动过滤断崖暴跌股与中期下降通道(MA60向下)", value=True)

    if st.button("⚡ 执行今日全市场扫描"):
        conn = sqlite3.connect(DB_NAME, timeout=15)
        genes = pd.read_sql("SELECT * FROM stock_genes", conn)
        conn.close()
        
        scan_bar = st.progress(0.0)
        status_text = st.empty()
        index_df, _ = get_data('000001')
        hits = []
        
        def real_scan_task(row):
            df, name = get_data(row['code'])
            if df is None or len(df) < 260: return None
            
            O, C, H, L, V = df['开盘'], df['收盘'], df['最高'], df['最低'], df['成交量']
            
            if "三维共振" in strategy_choice:
                volatility = calculate_volatility(df)
                if volatility > max_vol: return None
                momentum = calculate_momentum(df, index_df) if index_df is not None else 0
                if momentum * 100 < min_mom: return None
                
                best_ma_period = row['best_ma'] if pd.notna(row['best_ma']) else 60
                ema_soul = C.ewm(span=best_ma_period, adjust=False).mean()
                
                if C.iloc[-1] > ema_soul.iloc[-1] and C.iloc[-2] <= ema_soul.iloc[-2]:
                    v_avg = V.shift(1).rolling(20).mean()
                    if V.iloc[-1] > v_avg.iloc[-1] * vol_m:
                        return {"代码": row['code'], "名称": row['name'], "策略": "三维共振", "当前价": round(C.iloc[-1], 2)}
                        
            elif "跳空爆发" in strategy_choice:
                gap_pct = (O.iloc[-1] - C.iloc[-2]) / C.iloc[-2] * 100
                is_gap_up = (O.iloc[-1] > H.iloc[-2]) or (gap_pct > 2.0)
                is_solid_green = C.iloc[-1] > O.iloc[-1]
                v_avg = V.shift(1).rolling(20).mean()
                is_vol_surge = V.iloc[-1] >= v_avg.iloc[-1] * vol_m
                ma20 = C.rolling(20).mean()
                is_uptrend = C.iloc[-1] > ma20.iloc[-1]
                
                if is_gap_up and is_solid_green and is_vol_surge and is_uptrend:
                    return {
                        "代码": row['code'], "名称": row['name'], "形态标签": "💥 N字起爆+向上缺口", 
                        "当前价": round(C.iloc[-1], 2), "跳空幅度": f"{gap_pct:.2f}%", "量比": f"{(V.iloc[-1]/v_avg.iloc[-1]):.2f}x"
                    }
                    
            else: 
                dyn_params = {"reject_downtrend": reject_knife}
                is_spring, roc_val, sqz_score, vol_ratio, is_annual_bounce, is_double_cross = calculate_advanced_indicators(df, dyn_params)
                if is_spring:
                    tag = "🔴极致压缩"
                    if is_annual_bounce: tag += " + 🐉年线回踩"
                    if is_double_cross: tag += " + 🌟双金叉"
                    return {
                        "代码": row['code'], "名称": row['name'], "形态标签": tag, 
                        "当前价": round(C.iloc[-1], 2), "ROC(12)": f"{roc_val:.2f}%", "压缩评分": f"{sqz_score:.1f}分"
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

# ----------------- Tab 3: 时光机与特征快照 -----------------
with tabs[2]:
    st.markdown("### ⏳ 时光机：历史扫描与全量特征捕捉")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: lookback_days = st.number_input("扫描过去多少天？", min_value=5, max_value=100, value=30, step=5)
    with col_t2: max_hold = st.number_input("验证未来几天收益？", min_value=3, max_value=30, value=10, step=1)
    with col_t3: backtest_strategy = st.selectbox("要验证的策略内核", ["前瞻预判 (左侧)", "三维共振 (右侧)", "跳空爆发 (缺口)"])
    
    if st.button("🚀 启动全域时光机回测"):
        conn = sqlite3.connect(DB_NAME, timeout=15)
        genes = pd.read_sql("SELECT * FROM stock_genes", conn)
        conn.close()
        test_bar = st.progress(0.0)
        status_text = st.empty()
        index_df_full, _ = get_data('000001')
        hits = []
        
        def vectorized_backtest_task(row):
            df, name = get_data(row['code'])
            if df is None or len(df) < 260: return []
            
            df_calc = df.copy()
            if index_df_full is not None:
                df_calc = pd.merge(df_calc, index_df_full[['日期', '收盘']], on='日期', how='left', suffixes=('', '_idx'))
                idx_c = df_calc['收盘_idx'].ffill() 
                idx_ret_20 = idx_c.pct_change(20)
            else: idx_ret_20 = 0
                
            O, C, H, L, V = df_calc['开盘'], df_calc['收盘'], df_calc['最高'], df_calc['最低'], df_calc['成交量']
            
            ma20 = C.rolling(20).mean()
            ma60 = C.rolling(60).mean()
            
            ma20_slope_matrix = (ma20 - ma20.shift(10)) / ma20.shift(10)
            is_falling_knife_matrix = (ma20_slope_matrix < -0.02)
            ma60_slope_matrix = (ma60 - ma60.shift(20)) / ma60.shift(20)
            is_macro_downtrend_matrix = (ma60_slope_matrix < -0.01) & (C < ma60)
            reject_mask = is_falling_knife_matrix | is_macro_downtrend_matrix
            
            std20 = C.rolling(20).std()
            bbw = (4 * std20) / ma20
            bbw_min_120 = bbw.rolling(120).min()
            bbw_max_120 = bbw.rolling(120).max()
            sqz_score_matrix = ((bbw - bbw_min_120) / (bbw_max_120 - bbw_min_120 + 1e-9)) * 100
            squeeze_on = (sqz_score_matrix <= 0.15)
            
            exp1 = C.ewm(span=12, adjust=False).mean()
            exp2 = C.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            momentum_pulse = hist > hist.shift(1)
            macd_cross = (macd.shift(1) < signal.shift(1)) & (macd > signal)
            
            roc12 = (C.diff(12) / C.shift(12)) * 100
            roc_pulse = (roc12 > roc12.shift(1)) & (roc12 >= 2.5)
            roc_cross = (roc12.shift(1) < 0) & (roc12 > 0)
            double_cross_matrix = macd_cross & roc_cross
            
            price_stable = C > (C.shift(3) * 0.96)
            v_ma3 = V.rolling(3).mean()
            v_ma20 = V.rolling(20).mean()
            vol_ratio_matrix = V / v_ma20
            volume_dry_up = v_ma3.shift(1) < (v_ma20.shift(1) * 0.70)
            vol_pulse = (vol_ratio_matrix >= 1.7)
            
            ma250 = C.rolling(250).mean()
            dist_ma250_matrix = (C - ma250) / ma250 * 100 
            recent_low = C.rolling(5).min() * 0.985
            annual_touch = (recent_low >= ma250 * 0.98) & (recent_low <= ma250 * 1.05) & (C > ma250)
            
            left_signal = squeeze_on & momentum_pulse & roc_pulse & price_stable & (volume_dry_up | vol_pulse) & (~reject_mask)
            
            vol_array = C.pct_change().rolling(60).std() * np.sqrt(252)
            mom_array = C.pct_change(20) - idx_ret_20
            best_ma = row['best_ma'] if pd.notna(row['best_ma']) else 60
            ema = C.ewm(span=best_ma, adjust=False).mean()
            cross_up = (C > ema) & (C.shift(1) <= ema.shift(1))
            v_avg = V.shift(1).rolling(20).mean()
            vol_break = V > (v_avg * 1.5)
            right_signal = (vol_array <= 0.4) & (mom_array * 100 >= 5) & cross_up & vol_break
            
            gap_pct_matrix = (O - C.shift(1)) / C.shift(1) * 100
            gap_up_matrix = (O > H.shift(1)) | (gap_pct_matrix > 2.0)
            solid_yang_matrix = C > O
            gap_vol_surge_matrix = V >= (v_avg * 1.5)
            uptrend_matrix = C > ma20
            gap_signal = gap_up_matrix & solid_yang_matrix & gap_vol_surge_matrix & uptrend_matrix
            
            if "三维" in backtest_strategy: target_signal = right_signal
            elif "跳空" in backtest_strategy: target_signal = gap_signal
            else: target_signal = left_signal
            
            triggers = []
            start_idx = max(260, len(df_calc) - lookback_days)
            for i in range(start_idx, len(df_calc)):
                if target_signal.iloc[i]:
                    trigger_date = df_calc['日期'].iloc[i]
                    trigger_price = C.iloc[i]
                    
                    if "跳空" in backtest_strategy:
                        tag = "💥缺口爆发"
                    else:
                        tag = "🔴极致压缩" if "前瞻" in backtest_strategy else "🟢三维突破"
                        if annual_touch.iloc[i]: tag += "+🐉年线"
                        if double_cross_matrix.iloc[i]: tag += "+🌟双金叉"
                    
                    fwd_ret = {}
                    for h in range(1, max_hold + 1):
                        if i + h < len(df_calc): fwd_ret[f'Hold_{h}D'] = (C.iloc[i+h] - trigger_price) / trigger_price * 100
                        else: fwd_ret[f'Hold_{h}D'] = np.nan
                            
                    triggers.append({
                        "代码": row['code'], "名称": name, "形态": tag, "触发日期": trigger_date, "触发价": round(trigger_price, 2),
                        "起爆_Squeeze评分": round(sqz_score_matrix.iloc[i], 2), "起爆_ROC": round(roc12.iloc[i], 2),
                        "起爆_量比": round(vol_ratio_matrix.iloc[i], 2), "起爆_年线乖离(%)": round(dist_ma250_matrix.iloc[i], 2) if pd.notna(dist_ma250_matrix.iloc[i]) else 999,
                        **fwd_ret
                    })
            return triggers

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
            futs = [ex.submit(vectorized_backtest_task, r) for _, r in genes.iterrows()]
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: hits.extend(res)
                if len(genes) > 0 and i % 10 == 0: 
                    test_bar.progress((i+1)/len(genes))
                    status_text.text(f"时光机采集样本中... 进度: {i+1} / {len(genes)} | 🎯 收集信号: {len(hits)} 个")
                    
        test_bar.progress(1.0)
        status_text.text(f"回测完毕。生成 {len(hits)} 份历史样本。")
        if hits:
            res_df = pd.DataFrame(hits)
            csv_data = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 导出回测 CSV 数据包", data=csv_data, file_name="时光机_特征样本.csv", mime="text/csv")
            st.dataframe(res_df.sort_values(by="触发日期", ascending=False))
            st.balloons()
        else: st.info("该区间内无股票触发策略。")

# ----------------- Tab 4: 归因分析提炼 -----------------
with tabs[3]:
    st.markdown("### 🏆 数据归因与提炼")
    uploaded_file = st.file_uploader("📂 上传 [时光机_特征样本.csv]", type=['csv'])
    if uploaded_file is not None:
        try:
            res_df = pd.read_csv(uploaded_file)
            hold_cols = [col for col in res_df.columns if 'Hold_' in col]
            clean_df = res_df.copy()
            for col in hold_cols:
                clean_df[col] = clean_df[col].astype(str).str.replace('%', '', regex=False).str.replace('+', '', regex=False).replace('等待开奖', np.nan).replace('nan', np.nan).astype(float)
            
            clean_df['最大涨幅'] = clean_df[hold_cols].max(axis=1)
            clean_df['最大跌幅'] = clean_df[hold_cols].min(axis=1)
            
            if '起爆_Squeeze评分' in clean_df.columns:
                winners = clean_df[clean_df['最大涨幅'] >= 10.0]  
                losers = clean_df[(clean_df['最大跌幅'] <= -5.0) & (clean_df['最大涨幅'] < 5.0)]  
                col_w, col_l = st.columns(2)
                with col_w:
                    st.success(f"🏆 成功组 (最大涨幅>10%): 共 {len(winners)} 只")
                    if not winners.empty: st.dataframe(winners[['名称', '触发日期', '起爆_Squeeze评分', '起爆_ROC', '最大涨幅']])
                with col_l:
                    st.error(f"☠️ 失败组 (最大跌幅<-5%且未曾大涨): 共 {len(losers)} 只")
                    if not losers.empty: st.dataframe(losers[['名称', '触发日期', '起爆_Squeeze评分', '起爆_ROC', '最大跌幅']])
            else: st.error("您上传的 CSV 不包含起爆特征。")
        except Exception as e: st.error(f"分析失败: {e}")

# ----------------- Tab 5: 规律通道猎手 (地狱漏斗版) -----------------
with tabs[4]:
    st.markdown("### 📈 规律通道挖掘 (地狱级漏斗过滤)")
    
    scan_mode = st.radio("选择通道挖掘模式:", [
        "1. 标准单通道 (规律震荡的高抛低吸)", 
        "2. 🪆 双通道嵌套 (跌破小通道引发恐慌 + 踩中大通道提供支撑)"
    ], horizontal=True)
    
    st.markdown("---")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    if "双通道" in scan_mode:
        with col_c1: channel_dir = st.selectbox("大通道方向:", ["下降通道", "上升通道"])
        with col_c2: max_macro_p = st.slider("大通道寻轨上限 (天以内)", 60, 400, 180, 10)
        with col_c3: max_micro_p = st.slider("小通道寻轨上限 (天以内)", 20, 200, 100, 10)
    else:
        with col_c1: 
            channel_dir = st.selectbox("选择要寻找的通道方向:", ["下降通道", "上升通道"])
            max_channel_period = st.slider("通道寻轨上限 (天以内)", 30, 250, 90, 10)
        with col_c2: 
            # 【新增】震幅与触碰次数控制面板
            min_width = st.slider("通道最小震幅空间 (%)", 10, 60, 30, 5)
            min_touches = st.slider("单边至少触碰次数", 2, 8, 3, 1)
        with col_c3: 
            only_bottom = st.checkbox("🎯 仅显示刚跌到通道下轨附近的股票", value=True)
            # 【新增】收盘价严格约束
            strict_close = st.checkbox("🚧 收盘价严格在通道内 (允许盘中刺穿)", value=True)
        
    if st.button("🔍 开始执行动态寻优运算"):
        conn = sqlite3.connect(DB_NAME, timeout=15)
        genes = pd.read_sql("SELECT * FROM stock_genes", conn)
        conn.close()
        
        c_bar = st.progress(0.0)
        c_status = st.empty()
        c_hits = []
        
        def channel_scan_task(row):
            df, name = get_data(row['code'])
            if df is not None:
                if "双通道" in scan_mode:
                    res = detect_dual_channel(df, max_macro_p=max_macro_p, max_micro_p=max_micro_p, direction=channel_dir)
                else:
                    # 【应用全新严苛参数】
                    res = detect_regression_channel(df, max_period=max_channel_period, direction=channel_dir, 
                                                    require_bottom=only_bottom, min_width=min_width, 
                                                    min_touches=min_touches, strict_close=strict_close)
                
                if res:
                    res["代码"] = row['code']
                    res["名称"] = name
                    return res
            return None

        with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
            futs = [ex.submit(channel_scan_task, r) for _, r in genes.iterrows()]
            for i, f in enumerate(as_completed(futs)):
                res = f.result()
                if res: c_hits.append(res)
                if len(genes) > 0 and i % 10 == 0: 
                    c_bar.progress((i+1)/len(genes))
                    c_status.text(f"正在全域执行地狱级过滤... 进度: {i+1} / {len(genes)} | 🎯 发现极品: {len(c_hits)} 个")
                    
        c_bar.progress(1.0)
        c_status.text(f"扫描完毕！大浪淘沙，仅找到 {len(c_hits)} 个符合绝佳几何形态的标的。")
        
        if c_hits:
            st.balloons()
            if "双通道" in scan_mode:
                df_display = pd.DataFrame(c_hits)[["代码", "名称", "形态", "大通道方向", "大通道(实)", "小通道(实)", "当前价", "距大下轨(支撑)", "距小下轨(破位)"]]
                st.dataframe(df_display)
                st.success("💡 **实战真言**：买入后，止损设在大通道下轨下方2%处，盈亏比极高！")
            else:
                df_display = pd.DataFrame(c_hits)[["代码", "名称", "通道方向", "实际天数", "通道空间", "上轨触碰", "下轨触碰", "当前价", "距下轨(买点)"]]
                st.dataframe(df_display)
                st.info("💡 **波段操作指南**：这批结果已经过最严苛的空间与触碰次数筛选。距下轨越近，买点越完美。")
        else:
            st.warning("当前市场没找到符合这种复杂震荡特征的股票，尝试放宽计算的寻轨上限。")
