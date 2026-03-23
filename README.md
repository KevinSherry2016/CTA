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
MovingAverage_V3的可执行版本

改进方向：
1. 分sector做参数
2. 信号定义，即z_score
z = (price - ma) / rolling_vol
或 z = (price / ma - 1) / rolling_vol
或 (price - ma) / ATR
这三种都比“直接对价格做 z-score”更接近“离均值有多远”。
3. 趋势追踪/追踪回复是完全两种逻辑
4. 持仓问题，不能够rolling（window），这样会导致仓位天然增加。
5. 不要采用全样本最优，而是walk-forward。如果一个参数在多个窗口里都排20%才有效。
6. 
加趋势过滤或波动过滤
很多品种在不同 regime 下同一逻辑完全相反。可以加：
长周期趋势过滤：只在大趋势向上时做多侧
波动过滤：低波动不做，高波动才做
成交活跃过滤：低活跃品种不参与
7. 做截面标准化，例如：
截面 rank
截面 z-score
波动率等权
每个 sector 风险预算一致
