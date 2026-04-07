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
