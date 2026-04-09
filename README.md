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


和徐博交流：
pot过小，因为每日signal变化很大。尤其是股指期货和债券。
方法：
改成状态机，或者/vol时候把时间拉大20->60

黑色趋势比较强，能化次之，其余sector一般。
是否需要剔除效果很差的品种。
A.DCE, PF.ZCE, PG.DCE, CS.DCE, RM.ZCE

不同sector是否需要不同信号计算方式和参数？
容易overfitting

因为使用同一种开仓方式，所以如果T=1，则pnl趋势相近。

股指期货主力是12个月？

研究方向：
1. 再加别的因子，然后和均线因子一起组合
2. 在趋势基础上，加ADX过滤等（因子择时）
3. 反转信号如何做？短趋势是否也可以认为是反转？（差别只是周期不同）

例如：cu 长周期趋势，短周期反转


均线，动量，kdj，macd。
量价相关性
衍生出来的指标：流动性：涨跌幅/成交额（振幅/成交额），直接作为因子，或者作为衍生指标

只取特定情况下量\价

alpha101
google scholar

多因子的本质，是需要各类低相关性因子组合
而不是在一个因子上深入纠结

时序，截面策略（delta netural策略）
通常股指期货/债券或许会做成delta netural策略，因为相关性极高



TODO：
1. 读取20个factor意思
2. 优化，通过分sector后合并，或者z-score
3. 信号本身 vs 状态机
