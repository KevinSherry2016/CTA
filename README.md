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
增加了信号定义

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
