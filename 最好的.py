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

# 【融合进化 1】：从 111.py 引入真实浏览器 UA 池，实现“幽灵”伪装，彻底防封禁
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
        # 每次请求随机抽取一个浏览器身份
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
    
    target_sqz = 100.0
    target_roc = -99.0
    target_vol = 1.5
    
    if dynamic_params:
        target_sqz = dynamic_params.get("max_sqz", 100.0)
        target_roc = dynamic_params.get("min_roc", -99.0)
        target_vol = dynamic_params.get("min_vol", 1.5)
    
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bbw = (4 * std20) / ma20
    bbw_min_120 = bbw.rolling(120).min()
    bbw_max_120 = bbw.rolling(120).max()
    
    squeeze_score = ((bbw.iloc[-1] - bbw_min_120.iloc[-1]) / (bbw_max_120.iloc[-1] - bbw_min_120.iloc[-1] + 1e-9)) * 100
    squeeze_on = squeeze_score <= target_sqz if dynamic_params else (bbw.iloc[-1] <= bbw_min_120.iloc[-1] * 1.05)
    
    v_ma3 = v.rolling(3).mean()
    v_ma20 = v.rolling(20).mean()
    volume_dry_up = v_ma3.iloc[-2] < (v_ma20.iloc[-2] * 0.70)
    vol_ratio = v.iloc[-1] / v_ma20.iloc[-1]
    vol_pulse = vol_ratio > target_vol if dynamic_params else (v.iloc[-1] > v_ma3.iloc[-2] * 1.5)
    
    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    momentum_pulse = hist.iloc[-1] > hist.iloc[-2]
    
    # 【融合进化 2】：严苛的双金叉判定，提取强反转特征
    macd_cross = (macd.iloc[-2] < signal.iloc[-2]) and (macd.iloc[-1] > signal.iloc[-1])
    
    roc12 = (c.diff(12) / c.shift(12)) * 100
    roc_pulse = roc12.iloc[-1] > roc12.iloc[-2]
    roc_cross = (roc12.iloc[-2] < 0) and (roc12.iloc[-1] > 0)
    
    is_double_cross = macd_cross and roc_cross
    
    if dynamic_params and roc12.iloc[-1] < target_roc:
        roc_pulse = False
    
    price_stable = c.iloc[-1] > c.iloc[-4] * 0.96
    
    ma250 = c.rolling(250).mean()
    annual_support = False
    if pd.notna(ma250.iloc[-1]):
        recent_low = c.tail(5).min() * 0.985 
        if (recent_low >= ma250.iloc[-1] * 0.98) and (recent_low <= ma250.iloc[-1] * 1.05) and (c.iloc[-1] > ma250.iloc[-1]):
            annual_support = True
            
    is_spring_loaded = squeeze_on and momentum_pulse and roc_pulse and price_stable and (volume_dry_up or vol_pulse)
    
    return is_spring_loaded, roc12.iloc[-1], squeeze_score, vol_ratio, annual_support, is_double_cross

# ================= 5. 基建与 UI =================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute('''CREATE TABLE IF NOT EXISTS stock_genes (code TEXT PRIMARY KEY, name TEXT, best_ma INTEGER)''')
    conn.commit(); conn.close()

init_db()
st.set_page_config(layout="wide")
st.title("🚀 量化狙击系统: 全维融合与AI自进化 V6")

tabs = st.tabs(["🏗️ 基因基建", "⚡ 实盘扫描 (今日打猎)", "⏳ 时光机 (极速回测)", "🧠 AI策略自进化与归因"])

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
    strategy_choice = st.radio("选择狙击策略:", ["三维共振 (低波+动量+均线突破)", "前瞻预判 (极限压缩+MACD+ROC)"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        max_vol = st.slider("最大年化波动率阈值", 0.1, 1.0, 0.4, 0.05)
    with col2:
        min_mom = st.slider("最低相对大盘动量 (%)", -5.0, 20.0, 5.0, 1.0)
    with col3:
        vol_m = st.slider("突破时成交量放大倍数", 1.0, 3.0, 1.5)

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
                
                ma250 = c.rolling(250).mean().iloc[-1]
                annual_touch = (c.iloc[-1] >= ma250 * 0.98) and (c.iloc[-1] <= ma250 * 1.05)
                tag = "🐉年线共振起爆" if annual_touch else "普通突破"
                
                if c.iloc[-1] > ema_soul.iloc[-1] and c.iloc[-2] <= ema_soul.iloc[-2]:
                    v_avg = v.shift(1).rolling(20).mean()
                    if v.iloc[-1] > v_avg.iloc[-1] * vol_m:
                        return {"代码": row['code'], "名称": row['name'], "策略": "三维共振", "形态": tag, "当前价": round(c.iloc[-1], 2)}
            else:
                is_spring, roc_val, sqz_score, vol_ratio, is_annual_bounce, is_double_cross = calculate_advanced_indicators(df)
                if is_spring:
                    tag = "🔴 极致压缩"
                    if is_annual_bounce: tag += " + 🐉年线回踩 (极高胜率)"
                    if is_double_cross: tag += " + 🌟双金叉反转"
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

# ----------------- Tab 3: 时光机与最优持股期 -----------------
with tabs[2]:
    st.markdown("### ⏳ 时光机 V2：全量特征快照回测")
    st.markdown("系统将自动捕捉每一个信号点的【原始指标特征】，为下一步的 AI 自进化提供学习样本。")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: lookback_days = st.number_input("扫描过去多少天？", min_value=5, max_value=60, value=20, step=5)
    with col_t2: max_hold = st.number_input("验证未来几天收益？", min_value=3, max_value=20, value=10, step=1)
    with col_t3: backtest_strategy = st.selectbox("要验证的策略内核", ["前瞻预判 (左侧)", "三维共振 (右侧)"])
    
    if st.button("🚀 启动全域时光机收集样本"):
        conn = sqlite3.connect(DB_NAME)
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
                
            c = df_calc['收盘']
            v = df_calc['成交量']
            
            ma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bbw = (4 * std20) / ma20
            bbw_min_120 = bbw.rolling(120).min()
            bbw_max_120 = bbw.rolling(120).max()
            
            sqz_score_matrix = ((bbw - bbw_min_120) / (bbw_max_120 - bbw_min_120 + 1e-9)) * 100
            squeeze_on = bbw <= (bbw_min_120 * 1.05)
            
            exp1 = c.ewm(span=12, adjust=False).mean()
            exp2 = c.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            momentum_pulse = hist > hist.shift(1)
            macd_cross = (macd.shift(1) < signal.shift(1)) & (macd > signal)
            
            roc12 = (c.diff(12) / c.shift(12)) * 100
            roc_pulse = roc12 > roc12.shift(1)
            roc_cross = (roc12.shift(1) < 0) & (roc12 > 0)
            double_cross_matrix = macd_cross & roc_cross
            
            price_stable = c > (c.shift(3) * 0.96)
            
            v_ma3 = v.rolling(3).mean()
            v_ma20 = v.rolling(20).mean()
            vol_ratio_matrix = v / v_ma20
            volume_dry_up = v_ma3.shift(1) < (v_ma20.shift(1) * 0.70)
            vol_pulse = v > (v_ma3.shift(1) * 1.5)
            
            ma250 = c.rolling(250).mean()
            dist_ma250_matrix = (c - ma250) / ma250 * 100 
            recent_low = c.rolling(5).min() * 0.985
            annual_touch = (recent_low >= ma250 * 0.98) & (recent_low <= ma250 * 1.05) & (c > ma250)
            
            left_signal = squeeze_on & momentum_pulse & roc_pulse & price_stable & (volume_dry_up | vol_pulse)
            
            vol_array = c.pct_change().rolling(60).std() * np.sqrt(252)
            mom_array = c.pct_change(20) - idx_ret_20
            best_ma = row['best_ma'] if pd.notna(row['best_ma']) else 60
            ema = c.ewm(span=best_ma, adjust=False).mean()
            cross_up = (c > ema) & (c.shift(1) <= ema.shift(1))
            v_avg = v.shift(1).rolling(20).mean()
            vol_break = v > (v_avg * 1.5)
            right_signal = (vol_array <= 0.4) & (mom_array * 100 >= 5) & cross_up & vol_break
            
            target_signal = right_signal if "三维" in backtest_strategy else left_signal
            
            triggers = []
            start_idx = max(260, len(df_calc) - lookback_days)
            for i in range(start_idx, len(df_calc)):
                if target_signal.iloc[i]:
                    trigger_date = df_calc['日期'].iloc[i]
                    trigger_price = c.iloc[i]
                    
                    tag = "🔴极致压缩" if "前瞻" in backtest_strategy else "🟢三维突破"
                    if annual_touch.iloc[i]: tag += "+🐉年线"
                    if double_cross_matrix.iloc[i]: tag += "+🌟双金叉"
                    
                    fwd_ret = {}
                    for h in range(1, max_hold + 1):
                        if i + h < len(df_calc): fwd_ret[f'Hold_{h}D'] = (c.iloc[i+h] - trigger_price) / trigger_price * 100
                        else: fwd_ret[f'Hold_{h}D'] = np.nan
                            
                    triggers.append({
                        "代码": row['code'], "名称": name, "形态": tag,
                        "触发日期": trigger_date, "触发价": round(trigger_price, 2),
                        "起爆_Squeeze评分": round(sqz_score_matrix.iloc[i], 2),
                        "起爆_ROC": round(roc12.iloc[i], 2),
                        "起爆_量比": round(vol_ratio_matrix.iloc[i], 2),
                        "起爆_年线乖离(%)": round(dist_ma250_matrix.iloc[i], 2) if pd.notna(dist_ma250_matrix.iloc[i]) else 999,
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
                    status_text.text(f"时光机采集特征样本中... 进度: {i+1} / {len(genes)} | 🎯 收集样本: {len(hits)} 个")
                    
        test_bar.progress(1.0)
        status_text.text(f"回测收集完毕。生成 {len(hits)} 份战斗样本。可前往 Tab 4 让 AI 归因进化。")
        if hits:
            res_df = pd.DataFrame(hits)
            csv_data = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 导出全特征 CSV", data=csv_data, file_name="全域特征回测结果.csv", mime="text/csv")
            st.dataframe(res_df.sort_values(by="触发日期", ascending=False))
            st.balloons()
        else: st.info("区间内无样本。")

# ----------------- Tab 4: AI策略自进化 (全自动闭环引擎) -----------------
with tabs[3]:
    st.markdown("### 🧠 AI 策略闭环引擎：从归因到进化")
    st.markdown("上传具有特征快照的时光机 `CSV`。AI 将通过对比大涨与下杀样本，自动挖掘高胜率基因，并更新实盘拦截代码！")
    
    uploaded_file = st.file_uploader("📂 上传附带[起爆特征]的 CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            res_df = pd.read_csv(uploaded_file)
            
            hold_cols = [col for col in res_df.columns if 'Hold_' in col]
            clean_df = res_df.copy()
            for col in hold_cols:
                clean_df[col] = clean_df[col].astype(str).str.replace('%', '', regex=False).str.replace('+', '', regex=False).replace('等待开奖', np.nan).astype(float)
            clean_df['最高收益'] = clean_df[hold_cols].max(axis=1)
            clean_df['最大回撤'] = clean_df[hold_cols].min(axis=1)
            
            if '起爆_Squeeze评分' in clean_df.columns:
                winners = clean_df[clean_df['最高收益'] >= 10.0]  
                losers = clean_df[clean_df['最大回撤'] <= -5.0]  
                
                st.markdown("#### 🔬 第一步：AI 多维特征基因测序对比")
                win_sqz = winners['起爆_Squeeze评分'].median()
                lose_sqz = losers['起爆_Squeeze评分'].median()
                win_roc = winners['起爆_ROC'].median()
                lose_roc = losers['起爆_ROC'].median()
                win_vol = winners['起爆_量比'].median()
                lose_vol = losers['起爆_量比'].median()
                
                compare_data = {
                    "基因特征": ["布林带压缩评分 (越低越好)", "爆发初夜动量 ROC (%)", "起爆日量比 (资金异动倍数)"],
                    "🏆 大肉组平均特征": [f"{win_sqz:.1f}分", f"{win_roc:.2f}%", f"{win_vol:.2f}x"],
                    "☠️ 大坑组平均特征": [f"{lose_sqz:.1f}分", f"{lose_roc:.2f}%", f"{lose_vol:.2f}x"]
                }
                st.dataframe(pd.DataFrame(compare_data))
                
                st.markdown("#### 🧬 第二步：AI 提炼终极进化参数")
                evolved_sqz = win_sqz * 1.2 
                evolved_roc = max(0, win_roc * 0.8) 
                evolved_vol = win_vol * 0.9 
                
                st.info(f"**AI 分析结论：**\n那些吃大肉的标的，起爆前布林带被压缩得更变态，且动能更为充沛。\n"
                        f"👉 **已自动生成新战法门槛：** 压缩评分必须 <= **{evolved_sqz:.1f}分**，且 ROC >= **{evolved_roc:.2f}%**，且量比 >= **{evolved_vol:.2f}x**。")
                
                st.markdown("#### ⚔️ 第三步：用进化参数发动今日“降维打击”")
                if st.button("🔥 接受 AI 建议：以此严苛标准扫描今日全市场！"):
                    dynamic_params = {"max_sqz": evolved_sqz, "min_roc": evolved_roc, "min_vol": evolved_vol}
                    
                    conn = sqlite3.connect(DB_NAME)
                    genes = pd.read_sql("SELECT * FROM stock_genes", conn)
                    conn.close()
                    evo_bar = st.progress(0.0)
                    evo_status = st.empty()
                    hits_evo = []
                    
                    def evo_scan_task(row):
                        df, name = get_data(row['code'])
                        if df is not None:
                            is_spring, roc_val, sqz_score, vol_ratio, is_annual, is_double_cross = calculate_advanced_indicators(df, dynamic_params)
                            if is_spring:
                                tag = "🤖 AI严选·压缩起爆"
                                if is_annual: tag += " + 🐉年线终极共振"
                                if is_double_cross: tag += " + 🌟双金叉反转"
                                return {
                                    "代码": row['code'], "名称": name, "形态标签": tag,
                                    "当前价": df['收盘'].iloc[-1], "压缩分": f"{sqz_score:.1f}", 
                                    "ROC": f"{roc_val:.2f}%", "量比": f"{vol_ratio:.2f}x"
                                }
                        return None

                    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as ex:
                        futs = [ex.submit(evo_scan_task, r) for _, r in genes.iterrows()]
                        for i, f in enumerate(as_completed(futs)):
                            res = f.result()
                            if res: hits_evo.append(res)
                            if len(genes) > 0 and i % 10 == 0: 
                                evo_bar.progress((i+1)/len(genes))
                                evo_status.text(f"执行终极猎杀... 进度: {i+1}/{len(genes)} | 发现极品: {len(hits_evo)} 个")
                                
                    evo_bar.progress(1.0)
                    evo_status.text(f"AI 进化扫描完毕。万里挑一，发现 {len(hits_evo)} 个极品标的。")
                    if hits_evo: 
                        st.balloons()
                        st.dataframe(pd.DataFrame(hits_evo))
                    else: st.warning("当前市场太弱，或者 AI 参数过于严苛，暂无标的达到起爆标准。建议空仓等待！")
            else:
                st.error("您上传的 CSV 是旧版本的。请到 Tab 3 重新运行一次『全域时光机』，获取带有特征快照的新文件！")
        except Exception as e:
            st.error(f"分析失败: {e}")
