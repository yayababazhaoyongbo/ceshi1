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

# ================= 3. 平行双钢尺通道寻轨算法 (重构：绝对不切K线实体) =================
def detect_regression_channel(df, max_period=90, direction='下降通道', require_bottom=True, min_width=30, min_touches=3):
    """
    【全新重构】双钢尺动态包裹算法
    100% 保证上下边界线紧贴K线最外侧影线，没有任何实体或影线会被“切西瓜”斩断。
    """
    actual_max = min(max_period, len(df))
    if actual_max < 30: return None
    
    # 从最大周期开始动态收敛寻找
    for p in range(actual_max, 29, -5):
        df_slice = df.tail(p)
        x = np.arange(p)
        highs = df_slice['最高'].values
        lows = df_slice['最低'].values
        closes = df_slice['收盘'].values
        
        # 估算斜率寻优区间
        price_span = max(highs.max() - lows.min(), 0.01)
        max_slope = price_span / p
        
        # 构建旋转直尺的候选斜率池
        if direction == '下降通道':
            slopes = np.linspace(-2.0 * max_slope, -0.05 * max_slope, 150)
        else:
            slopes = np.linspace(0.05 * max_slope, 2.0 * max_slope, 150)
            
        best_p_slope = None
        best_p_score = -1
        best_p_width = 999999
        best_p_data = None
        
        for k in slopes:
            # 核心公式：寻找能将所有 K 线完美包络在内的截距
            diffs_high = highs - k * x
            diffs_low = lows - k * x
            
            b_u = np.max(diffs_high) # 绝对上限界
            b_d = np.min(diffs_low)  # 绝对下限界
            
            width = b_u - b_d
            if width <= 0: continue
            
            channel_width_pct = width / max(lows.min(), 0.01) * 100
            if channel_width_pct < min_width: continue
            
            # 容错厚度定义为通道总宽度的 3% (模拟手绘画线的粗细)
            epsilon = 0.03 * width
            
            # 计算有多少个真实的最高点/最低点贴在了钢尺边缘上
            up_touches = np.where(highs >= (k * x + b_u - epsilon))[0]
            dn_touches = np.where(lows <= (k * x + b_d + epsilon))[0]
            
            n_up = len(up_touches)
            n_dn = len(dn_touches)
            
            # 严格筛选：上下两边都必须经历过至少 min_touches 次的有效撞击
            if n_up < min_touches or n_dn < min_touches:
                continue
                
            # 防止撞击点挤在连续几天内（假通道）。触碰点首尾跨度必须超过通道长度的 25%
            up_spread = up_touches[-1] - up_touches[0]
            dn_spread = dn_touches[-1] - dn_touches[0]
            if up_spread < 0.25 * p or dn_spread < 0.25 * p:
                continue
                
            # 综合评分：撞击点越多越好，通道越窄（越紧凑贴合）越好
            score = n_up + n_dn
            if score > best_p_score or (score == best_p_score and channel_width_pct < best_p_width):
                best_p_score = score
                best_p_width = channel_width_pct
                best_p_slope = k
                best_p_data = {
                    "slope": k,
                    "b_u": b_u,
                    "b_d": b_d,
                    "channel_width": channel_width_pct,
                    "up_touches": n_up,
                    "dn_touches": n_dn,
                    "p": p
                }
                
        if best_p_data is not None:
            # 成功在当前周期寻找到完美包络通道！
            k = best_p_data["slope"]
            b_d = best_p_data["b_d"]
            current_close = closes[-1]
            current_lower_val = k * (p - 1) + b_d
            dist_to_lower = (current_close - current_lower_val) / max(current_lower_val, 0.01) * 100
            
            # 判定当前价格是否在底部的买入带
            if require_bottom and (dist_to_lower > 3.0 or dist_to_lower < -2.0):
                continue
                
            return {
                "通道方向": "↘️ 下降通道" if direction == '下降通道' else "↗️ 上升通道",
                "实际天数": f"{p}",
                "通道空间": f"{best_p_data['channel_width']:.1f}%",
                "上轨触碰": f"{best_p_data['up_touches']}次",
                "下轨触碰": f"{best_p_data['dn_touches']}次",
                "当前价": round(current_close, 2),
                "距下轨(买点)": f"{dist_to_lower:.2f}%",
                "slope": k,
                "b_u": best_p_data["b_u"],
                "b_d": b_d
            }
    return None

def detect_dual_channel(df, max_macro_p=180, max_micro_p=100, direction='下降通道'):
    """
    【升级版】双通道嵌套雷达
    """
    actual_mac = min(max_macro_p, len(df))
    actual_mic = min(max_micro_p, len(df))
    if actual_mac < 60: return None
    
    valid_macros = []
    # 寻找完美包裹大周期的钢尺
    for mac_p in range(actual_mac, 59, -10):
        res_mac = detect_regression_channel(df, max_period=mac_p, direction=direction, require_bottom=True, min_width=25, min_touches=3)
        if res_mac:
            valid_macros.append(res_mac)
            break
            
    if not valid_macros: return None
    mac_res = valid_macros[0]
    mac_p = int(mac_res["实际天数"])
    
    # 在其内部寻找发生破位的小钢尺通道
    for mic_p in range(actual_mic, 19, -5):
        if mic_p >= mac_p: continue
        res_mic = detect_regression_channel(df, max_period=mic_p, direction=direction, require_bottom=False, min_width=15, min_touches=2)
        if res_mic:
            # 检查小通道是否在今天被击穿了底边
            dist_to_mic = float(res_mic["距下轨(买点)"].replace('%', ''))
            if dist_to_mic < 0.5: # 价格已经跌破或粘在小下轨上
                return {
                    "形态": "🪆双通道破底翻(极限洗盘)",
                    "大通道方向": mac_res["通道方向"],
                    "大通道(实)": f"{mac_p}",
                    "小通道(实)": f"{mic_p}",
                    "当前价": mac_res["当前价"],
                    "距大下轨(支撑)": mac_res["距下轨(买点)"],
                    "距小下轨(破位)": f"{dist_to_mic:.2f}% (破底诱空)"
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

if not db_ready:
    st.error("云端存储环境初始化失败，请稍后刷新重试。")
    st.stop()

tabs = st.tabs(["🏗️ 基因基建", "⚡ 实盘扫描", "⏳ 时光机回测", "🏆 归因提炼", "📈 规律通道猎手", "📐 画线辅助(绝对点)"])

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

# ----------------- Tab 5: 规律通道猎手 (真实极值平移版) -----------------
with tabs[4]:
    st.markdown("### 📈 规律通道挖掘 (视觉级极点平移算法)")
    
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
            min_width = st.slider("通道最小震幅空间 (%)", 10, 60, 30, 5)
            # 【重要关联】：这个触碰次数现在也是决定通道画线位置的关键锚点！
            min_touches = st.slider("至少锚定第N个极值点 (建议3)", 2, 8, 3, 1)
        with col_c3: 
            only_bottom = st.checkbox("🎯 仅显示刚跌到通道下轨附近的股票", value=True)
            strict_close = st.checkbox("🚧 收盘价严格在通道盖子内 (只允许影线刺穿)", value=True)
        
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
                    c_status.text(f"正在全域执行极值平移算法... 进度: {i+1} / {len(genes)} | 🎯 发现极品: {len(c_hits)} 个")
                    
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
                st.info("💡 **操作指南**：这批结果已经过最严苛的空间与收盘价盖子筛选。请将表格中的【代码】和【实际天数】填入 Tab 6 提取绝对坐标！")
        else:
            st.warning("当前市场没找到符合这种复杂震荡特征的股票，尝试放宽计算的寻轨上限。")

# ----------------- Tab 6: 画线辅助(绝对点) -----------------
with tabs[5]:
    st.markdown("### 📐 通道画线辅助 (傻瓜式绝对坐标定位)")
    st.markdown("不用再去找哪一天最高、哪一天最低了！系统直接为您提供**通道起点的理论价格**和**通道终点的理论价格**。您只需用软件的【画线段工具】，将这两个点连起来，完美的通道瞬间呈现！")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        draw_code = st.text_input("输入股票代码 (如: 000001)")
    with col_d2:
        draw_period = st.number_input("输入 Tab 5 中的【实际天数】", min_value=20, max_value=400, value=60)
    with col_d3:
        draw_touches = st.number_input("输入 Tab 5 中的【至少锚定极值点】", min_value=2, max_value=8, value=3)

    if st.button("📍 一键提取【直线绝对坐标】"):
        if draw_code:
            df, name = get_data(draw_code)
            if df is not None and len(df) >= draw_period:
                df_slice = df.tail(int(draw_period)).copy()
                df_slice.reset_index(drop=True, inplace=True)

                x = np.arange(int(draw_period))
                y = df_slice['收盘'].values
                highs = df_slice['最高'].values
                lows = df_slice['最低'].values

                # 【终极同步】：采用最高价与最低价斜率均值来计算中轴斜率，完美解决“切西瓜”
                slope_high, _ = np.polyfit(x, highs, 1)
                slope_low, _ = np.polyfit(x, lows, 1)
                slope = (slope_high + slope_low) / 2
                
                intercept = np.mean(y) - slope * np.mean(x)
                reg_line = slope * x + intercept
                
                dist_high = highs - reg_line
                dist_low = reg_line - lows
                
                try:
                    upper_offset = np.sort(dist_high)[-int(draw_touches)]
                    lower_offset = np.sort(dist_low)[-int(draw_touches)]
                except:
                    st.error("计算锚点失败，K线数量可能不足。")
                    st.stop()

                upper_band = reg_line + upper_offset
                lower_band = reg_line - lower_offset

                # 提取直线的绝对起止坐标
                start_date = df_slice['日期'].iloc[0]
                end_date = df_slice['日期'].iloc[-1]
                
                up_start_val = upper_band[0]
                up_end_val = upper_band[-1]
                
                dn_start_val = lower_band[0]
                dn_end_val = lower_band[-1]

                st.success(f"✅ [{draw_code} {name}] 完美回归通道计算完成！")

                st.markdown("#### ✏️ 请在炒股软件中使用【画线段工具】，将下面的点连成直线：")
                
                # 美化输出卡片
                st.error(f"🔴 **画【上轨阻力线】**：\n\n"
                        f"👉 **第一步（找起点）**：鼠标移动到 **{start_date}** 这天的 K 线，将光标垂直移动到价格 **{up_start_val:.2f}** 处，点下左键。\n\n"
                        f"👉 **第二步（拉终点）**：鼠标拉拽到 **{end_date}** (即今天)，将光标垂直移动到价格 **{up_end_val:.2f}** 处，再次点击。")

                st.success(f"🟢 **画【下轨支撑线】**：\n\n"
                           f"👉 **第一步（找起点）**：鼠标移动到 **{start_date}** 这天的 K 线，将光标垂直移动到价格 **{dn_start_val:.2f}** 处，点下左键。\n\n"
                           f"👉 **第二步（拉终点）**：鼠标拉拽到 **{end_date}** (即今天)，将光标垂直移动到价格 **{dn_end_val:.2f}** 处，再次点击。")
                
                st.markdown("---")
                st.info("💡 **神奇的见证时刻**：当您画完这两条线后，您会发现它们不仅绝对平行，而且上轨正好死死压住了这期间的 N 个最高点，下轨刚好精准托住了 N 个最低点。这就是平移锚定算法的威力！")

            else:
                st.error("股票代码无效或该股上市时间不足，请检查输入！")
        else:
            st.warning("请输入股票代码！")
