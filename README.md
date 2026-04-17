因子库：
动量与趋势类因子（Trend & Momentum）
1. 收益率因子：
因子值 = 最近N天的收益率

2. 价格均线偏离率：
因子值 =价格/最近N天的价格均线 - 1

3. 双均线交叉
因子值 = shortma/longma -1

4. MACD
dif = short ma - long ma
dea = dif.ewa().mean
MACD = (dif - dea)*2
注：通常用到的最佳参数为12  26  9

5. 唐奇安通道位置
因子值 = （close - 最近N天close的最小值）/（最近N天close的最大值 - 最近N天close的最小值） - 0.5
注：通常用到的最佳参数为20

6. 布林带突破
如果：close > 最近N天close均值 + 2*std，做多
如果：close > 最近N天close均值 - 2*std，做空

7. 相对强弱指数RSI
delta = close.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=N).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=N).mean()
rs = gain / (loss + 1e-9)
rsi = 100 - (100 / (1 + rs))
signal = (rsi - 50) / 100
rs = N天内上涨幅度的均值/N天内下跌幅度的均值
rsi = 100 - (100/(1 + rs))，范围在[0,100]之间
因子值 = (rsi - 50)/100 ，范围在（-0.5,0.5）

波动率与风险类因子（Volatility & Risk）
1. 真实波动幅度
tr = max(high-low, abs(high - preclose), abs(low - preclose)).max
atr = tr/close.rolling().mean()
atr_ratio = atr / close.replace(0, np.nan)
atr_ratio_mean = atr_ratio.rolling(window=baseline_window, min_periods=20).mean()
atr_ratio_std = atr_ratio.rolling(window=baseline_window, min_periods=20).std()
zscore = (atr_ratio - atr_ratio_mean) / (atr_ratio_std + 1e-12)
因子值 = (-zscore).clip(-3, 3) / 3


2. 历史收益率波动率
ret = close.pct_change()
hv = ret.rolling().std()
baseline_window = 50
hv_mean = hv.rolling(window=baseline_window, min_periods=20).mean()
hv_std = hv.rolling(window=baseline_window, min_periods=20).std()
zscore = (hv - hv_mean) / (hv_std + 1e-12)
因子值 = (-zscore).clip(-3, 3) / 3

3. 日内振幅
amp = (high - low)/open或者(high - low)/close(t-1)
amp_mean_window = amp.rolling(window=N).mean()
baseline_window = 50
amp_baseline_mean = amp_mean_window.rolling(window=baseline_window, min_periods=20).mean()
amp_baseline_std = amp_mean_window.rolling(window=baseline_window, min_periods=20).std()
zscore = (amp_mean_window - amp_baseline_mean) / (amp_baseline_std + 1e-12)
signal = (-zscore).clip(-3, 3) / 3

4. 上下行波动率倾向
因子 = (上涨收益率的波动 - 下跌收益率的波动) / (上涨收益率 + 下跌收益率波动)

成交量与持仓量类因子（Volume & Open Interest）
1. 成交量/持仓量动量
因子值 = 成交量/最近N天成交量均值 -1 （或持仓量）
注：
在此基础上，添加方向条件用以过滤

2. 量价相关性
因子值 = N天return和成交量变化的correlation

3. 能量潮指标（OBV - On Balance Volume）
direction = np.sign(close.diff())
obv = (direction * volume).cumsum()
因子值 = obv / obv.rolling(window=N).mean() - 1
注：
用以衡量资金的净流进和流出

4. 持仓量变化率
因子值 = oi / oi.shift(N) - 1

截面与非对称性因子（Cross-sectional & Asymmetry）
1. 收益率偏度
因子值 = ret.rolling(window=N).skew()

2. 收益率峰度
因子值 = -ret.rolling(window=N).kurt()
注：
pandas中，计算的是超额峰度

3. 隔夜与日内收益率差异
IntradayRet：close / open -1
OvernightRet： open / close(t-1) - 1
因子值 = (IntradayRet - OvernightRet).rolling().mean()

微观结构演化
1. amihud 缺乏流动性指标
|return|/Volume，衡量单位成交量可以推动多大价格变化，反应流动性
因子值 = return.abs/volume后进行z-score

2. 买卖压力
(close - low) / (high - low)，越接近1，表示收盘越强势。
因子值 = (((close - low) / (high - low))-0.5)*2


没有明显涨跌意义，仅适合做过滤：
波动率与风险类因子（Volatility & Risk）中 1,2,3
成交量与持仓量类因子（Volume & Open Interest）中 1,4
截面与非对称性因子（Cross-sectional & Asymmetry）中 1，2
微观结构演化中1


收益率因子：
Ferrous 20，其余40，都使用状态机


MACD因子：
农产品、能化、黑色  100  z-score
有色、贵金属 80 状态机


OvernightVsIntraday因子
农产品：50  No z-score
能化  40  状态机
铁、有色、贵金属  20  状态机

思路：
1. 同一种信号，不同周期策略合成
2. 剔除一些难做的品种，只做主要品种
3. 大趋势 +  过滤（放大）

量价关系很典型：
放量上涨 + 持仓增加 -> 趋势强化
放量下跌 + 持仓增加 -> 空头趋势强化

黑色系
能源化工
贵金属
核心有色（铜、铝）
部分农产品（油脂油料、白糖、棉花）

核心思路：
用动量/均线做方向骨架，用成交量/持仓量做趋势确认，用波动率因子做环境识别和风险缩放，用微观因子做入场择时和持仓微调，再通过板块约束实现组合层面的稳定化。

注意：
成交量/持仓量在换月时的处理：
只在同一合约内计算
换月日及之后 3~5 天：主力量仓变化因子降权或禁用
