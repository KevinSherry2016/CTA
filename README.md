行情数据：
fetchDailyData：从接口获取每日所有品种行情数据
generateMainContract：按照规则组成主连合约
linkMainContract：将主连复权
checkMainContract：主力合约检查

MovingAverage_V1：
参数M、N分别表示长周期和短周期均线，T表示持仓周期
金叉时做多，死差时做空。平滑T日

MovingAverage_V2：
参数M分别表示周期均线，N表示持仓周期，z_open表示开仓阈值
价格进行z_score处理后，如果>z_open做多，如果<-z_open做空

MovingAverage_V3：
参数M分别表示周期均线，z_open表示开仓阈值，z_close表示平仓阈值，max_hold表示最长持仓天数
signal可以选择趋势追踪或者反转
计算z_score，在满足开仓条件后开仓，在满足平仓条件或者达到最长持仓天数时平仓

MovingAverage_V4：
参数FAST_M_LIST分别表示短周期均线，SLOW_M_LIST表示长周期均线，z_open表示开仓阈值，z_close表示平仓阈值
signal可以选择趋势追踪或者反转
计算z_score，在满足开仓条件后开仓，在满足平仓条件时平仓

MovingAverage_V4_1：
优化了参数
增加了信号定义（共计4种）
支持分sector/全品种回测
参数T表示将每日仓位平滑T日
最后生成的仓位除以vol

MovingAverage_V4_2：
最优sector，合并后得到最终版本
最后生成的仓位除以vol

MovingAverage_5：
在V4_1的基础上，删除z-open和z-close。使用信号强度作为仓位（z-score后）而不是状态机。

MovingAverage_5_1：
信号 = (fast_ma - slow_ma) / 收盘价滚动标准差，z-score作为开关

MovingAverage_5_2：
信号 = (fast_ma - slow_ma) / ATR，z-score作为开关


MovingAverage_5_3：
信号 = (收盘价 - slow_ma) / ATR，z-score作为开关

MovingAverage_5_4：
信号 = (slow_ma - slow_ma.shift(slp_k)) / ATR，z-score作为开关


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




