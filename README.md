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

Momentum_V1
计算最近N天的return随后标准化（z-score），然后在因子层面进行cross-section比较。仅交易TOPN
最后生成的仓位再除以vol


交流
5. 什么是kdj/rsi做动量？（每个指标都可以做趋势或者反转）
7. 反转具体怎么做？（和return的相关性）
11. 状态过滤因子，例如ADX/R-squared of trend regression/Choppiness Index/均线发散度/波动率分位数等，用于过滤震荡市(很多判断方法)
14. 因子的有效性、衰减（markout看延迟开仓衰减速度，和交易相关 CDF）
因子失效的情况：
第一：周期性失效
第二：机器学习，数据挖掘出的因子，不可解释的，如果pnl表现不好
15. 股指期货趋势策略常用的参数大概什么范围（一个月，一个季度,120天）


z-score 后变好：往往说明原信号在不同波动/不同 regime 下尺度漂移很大（非平稳），用 z-score 做了“自适应尺度校准”，减少了某些时期过度加仓/过度交易。
z-score 后变差：往往说明原信号的“绝对水平”本身有信息（例如趋势强度越大越可靠），你把它改成“相对异常”后，反而削弱了可交易强度。


稳定受益于 z-score：Energy（四种信号都明显提升）。
稳定受损：Bond、Precious（四种信号几乎都下降）。
混合型：StockIndex、Other、All（取决于信号定义）。


一、先看四种信号各自偏好什么市场结构
V5_1: (fast-slow)/波动率
偏“趋势强度归一化”，适合有持续单边段、且波动会阶段性放大的品种。
V5_2: (fast-slow)/ATR
和V5_1类似，但ATR对跳空和日内振幅更敏感，适合波动冲击更频繁的品种。
V5_3: (price-slow)/ATR
偏“偏离慢均线后的再定价”，更像抓中期偏离和回归-再趋势化切换。
V5_4: slow slope/ATR
偏“慢趋势斜率”，更稳但更钝，适合中长趋势清晰、噪声较高的场景。

二、为什么不同sector匹配不同信号
Energy 适合 V5_1 + z-score
能源常见趋势脉冲+波动扩张，V5_1直接刻画“快慢差强度”，z-score做尺度校准后能减少极端波动期的仓位失真。
Bond 适合 V5_1 noZ
债券趋势往往慢而平滑，绝对趋势幅度本身有信息。z-score把这种绝对幅度“去量纲”后，反而削弱信号。
Ferrous 适合 V5_3（noZ更优，Z可备选）
黑色链条有中期景气与库存周期，价格相对慢均线偏离的信息量大。V5_3高原宽说明不是靠某个点参数吃饭。
NonFerrous 适合 V5_2（稳健优先用Z）
有色受宏观和风险偏好共同驱动，波动冲击多。ATR归一化更合理，z-score后参数高原变宽，鲁棒性更强。
Precious 适合 noZ（V5_3稳健，V5_4高峰）
贵金属受宏观叙事驱动，趋势强度绝对值常有信息，z-score后显著受损，说明“相对异常”替代“绝对强度”不合适。
StockIndex 适合 V5_3 + z-score
股指常见“偏离慢均线后再趋势化”，V5_3更贴合，z-score能抑制不同时期波动尺度漂移。
Agriculture 适合 V5_3 noZ（V5_4可做进攻）
农产品受季节性和供需扰动影响，慢均线偏离型信号更稳。V5_4峰值高但高原窄，容易变成参数孤岛。
Other 适合 V5_3 + z-score
该组异质性高，过“尖”的高分方案容易是孤岛。V5_3虽然峰值不最高，但参数带更连续，更适合实盘。
All 用 V5_3/V5_1/V5_2 noZ 组合更合理
全品种聚合时，追求单点最高意义下降，跨品种稳定性更重要，高原宽方案更优。

三、参数为什么会长成现在这个样子
F普遍较短（5到20）
短均线负责捕捉启动，过长会错过拐点。
S在不同sector明显分层
Bond常偏长S（100到120）说明慢趋势主导。Energy/All多在40到100，反映中期趋势更有效。
ATR多固定在15仍有效
说明“波动基准窗口”在这批数据里不需要很细调，主要差异来自F/S与是否z-score。
SLP只在V5_4中固定值有效
V5_4更像结构性慢因子，参数自由度高了反而更容易过拟合。
T和VOL未形成差异
当前实验里它们基本固定，不是区分sector适配性的关键来源。

因子优化：
1. 分sector后，找出最近参数后按照风险平价合并
2. 信号本身 vs 状态机
3. z-score vs no z-score
4. /vol
5. 比较correlation


动量与趋势类因子（Trend & Momentum）
1. 收益率因子：
因子值 = 最近N天的收益率
注：存在bug，应该使用adjclose计算return

2. 价格均线偏离率：
因子值 =价格/最近N天的价格均线 - 1
注：存在bug，应该使用adjclose计算return

3. 双均线交叉
因子值 = shortma/longma -1
注：存在bug，应该使用adjclose计算

4. MACD
dif = short ma - long ma
dea = dif.ewa().mean
MACD = (dif - dea)*2
注：存在bug，应该使用adjclose计算，通常用到的最佳参数为12  26  9


5. 唐奇安通道位置
因子值 = （close - 最近N天close的最小值）/（最近N天close的最大值 - 最近N天close的最小值） - 0.5
注：存在bug，应该使用adjclose计算，通常用到的最佳参数为20

6. 布林带突破
如果：close > 最近N天close均值 + 2*std，做多
如果：close > 最近N天close均值 - 2*std，做空
注：存在bug，应该使用adjclose计算

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
注：存在bug，应该使用adjclose计算

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
