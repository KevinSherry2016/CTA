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


2. 历史收益率波动率
ret = close.pct_change()
hv = ret.rolling().std()

3. 日内振幅
amp = (high - low)/open或者(high - low)/close(t-1)

4. 上下行波动率倾向
因子 = (上涨收益率的波动 - 下跌收益率的波动) / (上涨收益率 + 下跌收益率波动)

5. VIXFix
highest_close = close.rolling(window=N, min_periods=1).max()
因子值 = (highest_close - low) / (highest_close + 1e-8) * 100

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

5. 价格持仓量分离
因子值 = - return * sign(oi/oi.rolling(window = N).mean - 1)

6. 蔡金资金流量指标(CMF)
Money Flow Multiplier = ((close -low) - (high - low))/ high - close
money_flow_volume = Money Flow Multiplier * volume
cmf = money_flow_volume.rolling(window=N, min_periods=1).sum() / (volume.rolling(window=N, min_periods=1).sum() + 1e-8) 

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


均值回归
1. 短期回转
因子值 = - N天return

2. 反向噪声
direction = (close-close.shift(N)).abs()
vol = close.shift(1).abs().sum().rolling(window = N)
ker = direction/(volatility + 1e-8)
trend_dir = sign = sign(close - close.shift(N))
因子值 = -trend_dir*(1 - ker)

注意：
ATR、HistoricalVolatility、IntradayAmplitude、VIXFix、IntradayAmihudquality始终为正

Ferrous
MovingAverageBias（State machine，N = 30） + OvernightVsIntraday（State machine，N = 20） + VolumeMomentum（RAW，N = 50）


NonFerrous（cu al）
MovingAveragebias（State machine，N = 40） + MACD（State machine，{'fast_n': 24, 'slow_n': 52, 'signal_n': 18}）

NonFerrous(others)
DualMACrossover（State machine,{'fast_n': 20, 'slow_n': 60}）+ BuyingSellingPressure（State machine，N = 30）

Energy
movingaveragebais（State machine，N = 40） + CMF（RAW，N=40）

Precious
DualMACrossover(20, 40) + DualMACrossover(20, 60) + BuyingSellingPressure(40)

Agriculture(油脂油料)
CMF(state machine 50)

Agriculture(软商品)
DonchianChannel_{State machine  N: 50}  


TODO：
