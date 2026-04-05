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

MovingAverage_V3_1：
分 sector 配置不同参数，合成一个组合信号

MovingAverage_V3_2：
替换 z_score 定义，支持 (price - ma) / vol、(price / ma - 1) / vol、(price - ma) / ATR

MovingAverage_V3_3：
将趋势追踪和均值回复拆成两套独立逻辑与参数网格

MovingAverage_V3_4：
在状态机持仓基础上增加最短持仓与再入场冷却期，避免持仓过于抖动

MovingAverage_V3_5：
使用 walk-forward 方式做参数选择，并记录每个样本外窗口采用的参数

MovingAverage_V3_6：
在信号后增加趋势过滤、波动过滤和流动性过滤

MovingAverage_V3_7：
对横截面信号做 rank / z-score 标准化，再做波动率等权和 sector 风险均衡


Momentum_V1
计算最近N天的return随后标准化（z-score），然后在因子层面进行cross-section比较。仅交易TOPN
最后生成的仓位再除以vol


交流
1. 数据：tushare（新浪财经）
3. 什么是单因子？（单/双）均线、动量、kdj、rsi(ok)
4. 正常的步骤是不是应该先简单因子，有效果了再加上各种辅助参数？（例如z-open，z-close）（ok）
5. 什么是kdj/rsi做动量？（每个指标都可以做趋势或者反转）
6. 之前你做股指期货，回测效果大概如何（单因子看的话）
7. 反转具体怎么做？（和return的相关性）
8. 如何风控？（投资组合）
10. 持仓类因子，遇到换月怎么办？（不知道）
11. 状态过滤因子，例如ADX/R-squared of trend regression/Choppiness Index/均线发散度/波动率分位数等，用于过滤震荡市(很多判断方法)
12. carry因子，例如back情况下，多头往往更顺（不考虑）
13. 做因子的时候，是每个因子单独做，还是在已有因子的基础上叠加？（例如趋势追踪因子叠加反转）(通常分开)
14. 因子的有效性、衰减（markout看延迟开仓衰减速度，和交易相关 CDF）
因子失效的情况：
第一：周期性失效
第二：机器学习，数据挖掘出的因子，不可解释的，如果pnl表现不好
15. 股指期货趋势策略常用的参数大概什么范围（一个月，一个季度）
