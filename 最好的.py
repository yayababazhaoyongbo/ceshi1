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
    
    # 【基于回测大数据的严苛进化阈值】
    # 之前是 target_sqz=100.0, target_roc=-99.0
    target_sqz = 0.15   # 绝不妥协：压缩评分必须小于0.15，过滤掉0.43的假压缩
    target_roc = 2.5    # 动量点火：12日动量必须达到2.5%以上，直接过滤掉0.55的软脚虾
    target_vol = 1.7    # 异动门槛：量比提高到1.7倍及格线
    
    if dynamic_params:
        target_sqz = dynamic_params.get("max_sqz", target_sqz)
        target_roc = dynamic_params.get("min_roc", target_roc)
        target_vol = dynamic_params.get("min_vol", target_vol)
    
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bbw = (4 * std20) / ma20
    bbw_min_120 = bbw.rolling(120).min()
    bbw_max_120 = bbw.rolling(120).max()
    
    squeeze_score = ((bbw.iloc[-1] - bbw_min_120.iloc[-1]) / (bbw_max_120.iloc[-1] - bbw_min_120.iloc[-1] + 1e-9)) * 100
    
    # 修改判定：抛弃旧的粗略判定，直接使用严苛的压缩评分
    squeeze_on = (squeeze_score <= target_sqz)
    
    v_ma3 = v.rolling(3).mean()
    v_ma20 = v.rolling(20).mean()
    volume_dry_up = v_ma3.iloc[-2] < (v_ma20.iloc[-2] * 0.70)
    vol_ratio = v.iloc[-1] / v_ma20.iloc[-1]
    
    # 放量必须超过严格阈值
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
    
    # 【核心杀招】：强制过滤掉点火初速度达不到要求的杂鱼
    if roc12.iloc[-1] < target_roc:
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
st.title("🚀 量化狙击系统: 人机协作归因版")

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
                is_spring, roc_val, sqz_score, vol_ratio, is_annual_bounce, is_double_cross = calculate_advanced_indicators(df)
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

# ----------------- Tab 3: 时光机与特征快照 -----------------
with tabs[2]:
    st.markdown("### ⏳ 时光机：历史扫描与全量特征捕捉")
    st.markdown("一次性扫描过去 N 天触发的信号，并将起爆那一刻的指标特征（压缩分、ROC等）拍照留存，方便下一步归因。")
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: lookback_days = st.number_input("扫描过去多少天？", min_value=5, max_value=100, value=30, step=5)
    with col_t2: max_hold = st.number_input("验证未来几天收益？", min_value=3, max_value=30, value=10, step=1)
    with col_t3: backtest_strategy = st.selectbox("要验证的策略内核", ["前瞻预判 (左侧)", "三维共振 (右侧)"])
    
    if st.button("🚀 启动全域时光机回测"):
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
                    status_text.text(f"时光机采集样本中... 进度: {i+1} / {len(genes)} | 🎯 收集信号: {len(hits)} 个")
                    
        test_bar.progress(1.0)
        status_text.text(f"回测完毕。生成 {len(hits)} 份历史样本。请下载 CSV 去 Tab 4 归因分析！")
        if hits:
            res_df = pd.DataFrame(hits)
            csv_data = res_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(label="📥 导出回测 CSV 数据包", data=csv_data, file_name="时光机_特征样本.csv", mime="text/csv")
            st.dataframe(res_df.sort_values(by="触发日期", ascending=False))
            st.balloons()
        else: st.info("该区间内无股票触发策略。")

# ----------------- Tab 4: 归因分析提炼 (人机协作核心) -----------------
with tabs[3]:
    st.markdown("### 🏆 数据归因与提炼 (专为喂给大模型设计)")
    st.markdown("上传 Tab 3 的 CSV 文件，系统会提炼出高收益股和下跌股的核心基因差异，并生成一段**专供您复制给 AI 助手**的话术！")
    
    uploaded_file = st.file_uploader("📂 上传 [时光机_特征样本.csv]", type=['csv'])
    
    if uploaded_file is not None:
        try:
            res_df = pd.read_csv(uploaded_file)
            
            # 清洗未来收益数据
            hold_cols = [col for col in res_df.columns if 'Hold_' in col]
            clean_df = res_df.copy()
            for col in hold_cols:
                clean_df[col] = clean_df[col].astype(str).str.replace('%', '', regex=False).str.replace('+', '', regex=False).replace('等待开奖', np.nan).replace('nan', np.nan).astype(float)
            
            clean_df['最大涨幅'] = clean_df[hold_cols].max(axis=1)
            clean_df['最大跌幅'] = clean_df[hold_cols].min(axis=1)
            
            if '起爆_Squeeze评分' in clean_df.columns:
                # 分离大肉组和吃面组
                winners = clean_df[clean_df['最大涨幅'] >= 10.0]  
                losers = clean_df[(clean_df['最大跌幅'] <= -5.0) & (clean_df['最大涨幅'] < 5.0)]  
                
                col_w, col_l = st.columns(2)
                with col_w:
                    st.success(f"🏆 成功组 (最大涨幅>10%): 共 {len(winners)} 只")
                    if not winners.empty:
                        st.dataframe(winners[['名称', '触发日期', '起爆_Squeeze评分', '起爆_ROC', '最大涨幅']])
                with col_l:
                    st.error(f"☠️ 失败组 (最大跌幅<-5%且未曾大涨): 共 {len(losers)} 只")
                    if not losers.empty:
                        st.dataframe(losers[['名称', '触发日期', '起爆_Squeeze评分', '起爆_ROC', '最大跌幅']])
                
                st.divider()
                st.markdown("#### 🤖 专属 AI 沟通报告")
                st.write("点击输入框即可全选复制，直接发给对话框里的 AI（比如我），让我为您升级代码！")
                
                win_sqz = winners['起爆_Squeeze评分'].median() if not winners.empty else 0
                win_roc = winners['起爆_ROC'].median() if not winners.empty else 0
                win_vol = winners['起爆_量比'].median() if not winners.empty else 0
                
                lose_sqz = losers['起爆_Squeeze评分'].median() if not losers.empty else 0
                lose_roc = losers['起爆_ROC'].median() if not losers.empty else 0
                lose_vol = losers['起爆_量比'].median() if not losers.empty else 0
                
                report_text = (
                    f"你好！我刚刚用系统对历史进行了回测归因，得到了以下量化特征对比数据，请帮我分析并修改Python代码的过滤参数：\n\n"
                    f"【成功起爆的股票特征 (涨幅>10%)】\n"
                    f"- 启动前布林带压缩评分中位数：{win_sqz:.2f}分\n"
                    f"- 启动前 ROC(12) 动量中位数：{win_roc:.2f}%\n"
                    f"- 启动当量比中位数：{win_vol:.2f}x\n\n"
                    f"【失败被套的股票特征 (跌幅>-5%且未涨)】\n"
                    f"- 启动前布林带压缩评分中位数：{lose_sqz:.2f}分\n"
                    f"- 启动前 ROC(12) 动量中位数：{lose_roc:.2f}%\n"
                    f"- 启动当量比中位数：{lose_vol:.2f}x\n\n"
                    f"我的诉求：\n"
                    f"1. 请根据上述数据对比，帮我找出成功组和失败组最明显的区分点。\n"
                    f"2. 请帮我给出更严格的 `calculate_advanced_indicators` 函数代码替换方案，以便实盘扫描时滤除那些失败组特征的杂鱼股票。"
                )
                
                st.text_area("📋 报告内容 (请复制发送给AI)：", value=report_text, height=350)
                
            else:
                st.error("您上传的 CSV 不包含起爆特征。请去 Tab 3 跑一次最新的时光机并下载 CSV！")
        except Exception as e:
            st.error(f"分析失败: {e}")
