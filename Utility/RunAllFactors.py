import os
import numpy as np
import pandas as pd

# ==========================================
# ---------------- 数据与路径配置 ----------------
# ==========================================
MARKET_DATA_PATH = './main_contract/'  # 存放主力连续合约日线数据的目录
INFO_PATH = './Info.csv'               # 品种信息文件所在路径，包含品种代码和所属板块等
FINAL_VOL_WINDOW = 20                  # 计算品种日收益率波动率时使用的滚动窗口大小（常用于头寸按波动率缩放）
OUTPUT_DIR = './Result/'

# ==========================================
# -------------- 加载与处理基础数据 --------------
# ==========================================
# 1. 读取合约基本信息
info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
# 2. 生成 板块名 -> [品种代码列表] 的映射字典，方便后续按板块进行因子测试
sector_map = info.groupby('sector')['ts_code'].apply(list).to_dict()
# 3. 定义需要排除回测的非商品板块类别
EXCLUDE_SECTORS = ['StockIndex', 'Bond', 'Other', 'Others']

# 4. 将所有有效合约的日线行情数据一次性读取到内存字典中，以提高后续回测遍历的速度
data = {}
for ts_code in info['ts_code']:
    fp = os.path.join(MARKET_DATA_PATH, f'{ts_code}.csv')
    if not os.path.exists(fp): continue
    # 读取行情CSV，统一将交易日期('trade_date')设置为字符串索引
    df = pd.read_csv(fp, dtype={'trade_date': str}).set_index('trade_date')
    # 遍历常见的行情数据列，强制转换为数值类型，无法转换的变为NaN，防止数据格式异常
    for col in ['open', 'high', 'low', 'close', 'adj_close', 'adj_open', 'adj_high', 'adj_low', 'vol', 'oi']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    data[ts_code] = df

# ==========================================
# --------------- 信号计算辅助函数 ---------------
# ==========================================
def rolling_zscore(series, window=60):
    """
    计算时间序列的滚动横截面 Z-Score 标准化值
    :param series: 输入的时间序列 (pd.Series)
    :param window: 滚动窗口大小 (默认60)
    :return: 经过标准化处理的序列
    """
    min_p = max(window // 2, 1) # 至少包含窗口的一半数据才计算
    mu = series.rolling(window, min_periods=min_p).mean()
    sigma = series.rolling(window, min_periods=min_p).std().replace(0, np.nan) # 防止标准差为0导致的被除零报错
    return (series - mu) / sigma

def MACD_signal(close, N):
    """
    计算MACD柱状图信号。
    :param close: 收盘价序列
    :param N: 慢线的时间窗口周期
    :return: MACD柱状图数值
    """
    short = max(3, int(N/2)) # 自动确定快线周期，这里设定为慢线的一半
    dif = close.ewm(span=short).mean() - close.ewm(span=N).mean()  # DIF：短期EMA减去长期EMA
    dea = dif.ewm(span=max(3, int(short/2))).mean()                # DEA：对DIF再次进行平滑移动平均
    return (dif - dea) * 2 # MACD 柱状图 = (DIF - DEA) * 2

def RSI_signal(close, N):
    """
    计算RSI相对强弱指标信号。
    :param close: 收盘价序列
    :param N: 滚动时间窗口周期
    :return: 经过归一化平移的RSI信号 (分布在 -0.5 到 0.5 左右)
    """
    delta = close.diff() # 计算每日收盘价变化幅度
    # 分离上涨收益(gain)和下跌收益(loss的绝对值)
    gain = delta.where(delta > 0, 0).rolling(window=N, min_periods=max(1, N//2)).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=N, min_periods=max(1, N//2)).mean()
    rs = gain / (loss + 1e-9) # 相对强度 RS，分母补充小量防止除以零
    rsi = 100 - (100 / (1 + rs)) # 标准的 RSI 计算公式
    return (rsi - 50) / 100      # 减去50并除以100进行中心化平移，产生正负多空对标信号

def get_signal(df, factor_name, N):
    """
    根据给定的因子名称和参数窗口N，计算对应品种在全部时间段的因子信号值。
    :param df: 该品种的日线数据 DataFrame
    :param factor_name: 因子名称字符串
    :param N: 计算因子所用的回溯周期参数
    :return: 返回含有时序原始信号值的 pd.Series
    """
    # 提取复权和非复权的行情基础序列
    close = df['adj_close']
    open_ = df['adj_open']
    high = df['adj_high']
    low = df['adj_low']
    # 如果数据集中没有成交量或持仓量(某些数据源可能缺失)，默认使用常数1进行填充避免计算错误
    vol = df.get('vol', pd.Series(1, index=close.index))
    oi = df.get('oi', pd.Series(1, index=close.index))
    pre_close = close.shift(1) # 获取昨收盘价
    
    # ================= 趋势类因子 =================
    # 1. 时序动量因子 (N日历史收益率大小作为多空信号强度)
    if factor_name == 'TimeSeriesMomentum': return close.pct_change(N).fillna(0)
    # 2. 价格均线偏离率 (乖离率 Bias: 当前价格相对N日均线的偏离幅度)
    elif factor_name == 'MovingAverageBias': return (close / close.rolling(N).mean() - 1).fillna(0)
    # 3. 双均线交叉 (短均线相对长均线的偏离度，短均线周期取长周期一半)
    elif factor_name == 'DualMACrossover': return (close.rolling(max(1, int(N/2))).mean() / close.rolling(N).mean() - 1).fillna(0)
    # 4. MACD 因子 (调用辅助函数)
    elif factor_name == 'MACD': return MACD_signal(close, N).fillna(0)
    # 5. 唐奇安通道位置 (当前价格在过去N日最高价与最低价构成的波动通道中所处的相对位置，减去0.5实现中心化)
    elif factor_name == 'DonchianChannel':
        c_min = low.rolling(N).min()
        c_max = high.rolling(N).max()
        return ((close - c_min) / (c_max - c_min + 1e-9) - 0.5).fillna(0)
    # 6. 布林带突破 (计算当前价格相对N日均线的偏离，再除以N日标准差，衡量突破统计区间的程度)
    elif factor_name == 'BollingerBands':
        mu = close.rolling(N).mean()
        std = close.rolling(N).std().replace(0, 1e-9)
        return ((close - mu) / std).fillna(0)
    # 7. 相对强弱指数RSI (过去N日上涨动能占总波动的比例)
    elif factor_name == 'RSI': return RSI_signal(close, N).fillna(0)
    
    # ================= 波动率与风险类因子 =================
    # 8. 真实波动幅度 (当前价格真实波动率的均值，取负值以捕捉高波状态下的均值回复/空头信号)
    elif factor_name == 'ATR':
        tr1 = high - low                                    # 当日高低价差
        tr2 = (high - pre_close).abs()                      # 当日高点与昨日收盘价跳空缺口
        tr3 = (low - pre_close).abs()                       # 当日低点与昨日收盘价跳空缺口
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) # 结合上述三者取最大即为绝对真实波幅
        atr_ratio = tr.rolling(N).mean() / (close+1e-9)     # 计算波动幅度占比
        return -rolling_zscore(atr_ratio, 50).fillna(0)
    # 9. 历史收益率波动率 (日收益率的N日滚动标准差，高波状态同样用负号转多空)
    elif factor_name == 'HistoricalVolatility': 
        hv = close.pct_change().rolling(N).std()
        return -rolling_zscore(hv, 50).fillna(0)
    # 10. 日内振幅 (日内高低点间距，反映日内多空博弈激烈程度)
    elif factor_name == 'IntradayAmplitude': 
        amp = (high - low) / (open_ + 1e-9)
        return -rolling_zscore(amp.rolling(N).mean(), 50).fillna(0)
    # 11. 上下行波动率倾向 (上行期波动与下行期波动的相对比值，捕捉潜在非对称风险)
    elif factor_name == 'DownsideUpsideVolatility':
        ret = close.pct_change()
        up_v = ret.where(ret > 0, 0).rolling(N).std() # 取上涨日的波动序列计算标准差
        dn_v = ret.where(ret < 0, 0).rolling(N).std() # 取下跌日的波动序列计算标准差
        return ((up_v - dn_v) / (up_v + dn_v + 1e-9)).fillna(0)
        
    # ================= 成交量与持仓量类 =================
    # 12. 成交量动量 (当前成交量对于N日平均成交量的偏离，放量或缩量)
    elif factor_name == 'VolumeMomentum': return (vol / (vol.rolling(N).mean() + 1e-9) - 1).fillna(0)
    # 13. 量价相关性 (过去N日当前价格涨幅和成交量变化率的滚动相关系数)
    elif factor_name == 'PriceVolumeCorrelation': return close.pct_change().rolling(N).corr(vol.pct_change()).fillna(0)
    # 14. 能量潮指标 OBV (通过价格当天涨跌的符号作为权重系数，累加成交量，反映资金多空买卖净流向)
    elif factor_name == 'OBV':
        obv = (np.sign(close.diff()) * vol).cumsum()
        return (obv / (obv.rolling(N).mean() + 1e-9) - 1).fillna(0)
    # 15. 持仓量变化率 (当前持仓量相对于N日前历史持仓量的变化百分比)
    elif factor_name == 'OpenInterestROC': return (oi / oi.shift(N) - 1).fillna(0)
    
    # ================= 截面与非对称性因子 =================
    # 16. 收益率偏度 (收益率分布的三阶矩，偏度异常通常代表存在崩盘风险或正尾部肥厚)
    elif factor_name == 'Skewness': return close.pct_change().rolling(N).skew().fillna(0)
    # 17. 收益率峰度 (收益率分布的四阶矩，反映极端行情的概率密度特征，由于极值易反转取负值)
    elif factor_name == 'Kurtosis': return -close.pct_change().rolling(N).kurt().fillna(0)
    # 18. 隔夜与日内收益率差异 (通过开盘跳空产生的隔夜收益与日终波动的日内收益两者的价差，捕捉某些特有交易习惯)
    elif factor_name == 'OvernightVsIntraday':
        intraday = close / (open_ + 1e-9) - 1          # 日内真实收益率 (收盘相对开盘)
        overnight = open_ / (pre_close + 1e-9) - 1     # 隔夜跳空收益率 (开盘相对昨收)
        return (intraday - overnight).rolling(N).mean().fillna(0)
        
    # ================= 微观结构演变 =================
    # 19. Amihud 缺乏流动性指标 (计算单位成交量能够驱动价格变动的绝对幅度，数值越大说明流动性越差)
    elif factor_name == 'AmihudIlliquidity': 
        amihud = close.pct_change().abs() / (vol + 1e-9)
        return -rolling_zscore(amihud, N).fillna(0)
    # 20. 买卖压力 (衡量收盘价是在整个日间最高最低价区间的顶部还是底部收盘，反映日终阶段哪一方掌控了最后的定价权)
    elif factor_name == 'BuyingSellingPressure':
        return (((close - low) / (high - low + 1e-9) - 0.5) * 2).rolling(N).mean().fillna(0)
        
    # 对于未在上述因子列表之内的无效名称，统一返回全零信号(空仓状态)
    return pd.Series(0, index=close.index)

# 需要进行多空信号测试的因子列表汇总
FACTORS = [
    'TimeSeriesMomentum', 'MovingAverageBias', 'DualMACrossover', 'MACD', 'DonchianChannel', 'BollingerBands', 'RSI',
    'ATR', 'HistoricalVolatility', 'IntradayAmplitude', 'DownsideUpsideVolatility',
    'VolumeMomentum', 'PriceVolumeCorrelation', 'OBV', 'OpenInterestROC',
    'Skewness', 'Kurtosis', 'OvernightVsIntraday',
    'AmihudIlliquidity', 'BuyingSellingPressure'
]

# 在进行因子搜索时评估的观察窗口参数N的候选测试列表
N_LIST = [5, 10, 20, 40, 60]

# 空列表用于收集并存储每一个因子在不同板块测试下的最终夏普和汇总结果
results = []

# ==========================================
# ------------- 因子各板块遍历回测引擎 -------------
# ==========================================
for factor in FACTORS:
    print(f"Testing factor: {factor}")
    # 将商品市场数据按各个不同板块进行隔离测试并最终分开评估（比如区分农产品和黑色系等）
    for sector, ts_codes in sector_map.items():
        if sector in EXCLUDE_SECTORS: continue # 若该板块在排除列表中则直接跳过
        # 筛选出属于当前板块而且已经成功在数据字典中读取到的品种
        valid_symbols = [c for c in ts_codes if c in data]
        if not valid_symbols: continue
            
        # 记录当前板块在最佳参数设置下的参数N以及取得的夏普比率表现
        best_n = None
        best_sharpes = {'No zscore': -999, 'zscore': -999, 'State Machine': -999}
        max_metric = -999
        
        # 对于给定的N的参数序列进行循环，相当于简单的网格参数寻优
        for N in N_LIST:
            pnl_noz, pnl_z, pnl_sm = [], [], [] # 分别记录原始信号、Z-Score标准化信号、符号信号生成的每日盈亏序列
            
            # 对该板块下的每一个单品种分别生成对应的信号仓位
            for ts_code in valid_symbols:
                df = data[ts_code]
                daily_ret = df['adj_close'].pct_change(fill_method=None) # 计算日度对数或简单收益率
                # 使用20日滚动收益率波动率对风险平价缩放因子进行度量运算
                vol = daily_ret.rolling(FINAL_VOL_WINDOW, min_periods=1).std().replace(0, np.nan)
                
                # 获取该品种利用参数N提取的当日的横截面/时序原始特征信号
                raw_sig = get_signal(df, factor, N)
                
                # 模式1：直接采用原始信号进行缩放，不进行后续处理直接作为头寸配置信号
                pos_noz = raw_sig / vol
                
                # 模式2：将该原始信号进行长周期（如60日）时序标准化(Z-Score)
                z_sig = rolling_zscore(raw_sig, 60)
                pos_z = z_sig / vol
                
                # 模式3：对于信号使用二分类处理函数（仅仅考虑交易方向，而忽视强弱）
                sm_sig = np.sign(raw_sig)
                pos_sm = sm_sig / vol
                
                # 将信号往后滞推一天（shift）再与第二天的实际行情收益率相乘，防止未来函数泄露导致结果脱离现实
                pnl_noz.append(pos_noz.shift(1) * daily_ret)
                pnl_z.append(pos_z.shift(1) * daily_ret)
                pnl_sm.append(pos_sm.shift(1) * daily_ret)
                
            if pnl_noz:
                # 按照日期（按行）分别累加每一个品种当天的投资收益，此时不考虑跨板块头寸配比问题
                df_noz = pd.concat(pnl_noz, axis=1).sum(axis=1)
                df_z = pd.concat(pnl_z, axis=1).sum(axis=1)
                df_sm = pd.concat(pnl_sm, axis=1).sum(axis=1)
                
                # 利用日Pnl评估多策略的组合级测试年化夏普比率指标(交易天数设为252)
                sr_noz = df_noz.mean() / df_noz.std() * np.sqrt(252) if df_noz.std() > 0 else 0
                sr_z = df_z.mean() / df_z.std() * np.sqrt(252) if df_z.std() > 0 else 0
                sr_sm = df_sm.mean() / df_sm.std() * np.sqrt(252) if df_sm.std() > 0 else 0
                
                # Check parameter robustness (max parameter using 'No zscore' sr_noz, or default)
                # 以原味信号(No zscore)的表现作为判定参数优势与否的最主要评判标准，更新极值信息变量状态和缓存数据结构
                if sr_noz > max_metric:
                    max_metric = sr_noz
                    best_n = N
                    best_sharpes = {'No zscore': sr_noz, 'zscore': sr_z, 'State Machine': sr_sm}
                
        # 保存具有最大原始信号夏普的对应的一组测试因子综合结果列表
        if best_n is not None:
            results.append({
                'factor': factor,
                'sector': sector,
                'zscore': round(best_sharpes['zscore'], 4),               # 基于标准信号的夏普结果
                'No zscore': round(best_sharpes['No zscore'], 4),         # 原始信号表现
                'State Machine': round(best_sharpes['State Machine'], 4), # 纯符号表现
                'parameters': f'N={best_n}'
            })

# -------- 报告产生和报表文件归档存储阶段 --------
res_df = pd.DataFrame(results) # 转化为易于理解输出的数据帧结构
res_df.to_csv(OUTPUT_DIR + 'AllFactors_Result.csv', index=False) # 落盘保存最终所有的因子和对应最佳版块记录分析
print(res_df.head(20))
