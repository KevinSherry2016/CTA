# 因子清单
| 因子类型 | 因子名称 | 因子计算公式 | 注意事项 |
| 截面与非对称性 | 收益率峰度 | `-ret.rolling(window=N).kurt()`|`kurt()` 为超额峰度|
| 截面与非对称性 | 日内收益 vs 隔夜收益差异 | ``intraday_ret = close/(open+1e-9)-1`，`overnight_ret = open/(close.shift(1)+1e-9)-1`,`(intraday_ret - overnight_ret).rolling(window=N).mean()`|衡量“收益来源结构”，不是涨跌方向|
| 截面与非对称性 | 收益率偏度 | `ret.rolling(window=N).skew()`||
| 均值回归 | 反向噪声| `trend_dir = sign(close-close.shift(N))`，`direction = abs(close-close.shift(N))`，`volatility = rolling_sum(abs(close.diff()))`,`ker = direction/(volatility+1e-8)`,`-trend_dir * (1 - ker)` ||
| 均值回归 | 滚动ZScore反转 | `ma = close.rolling(N).mean()`，`std = close.rolling(N).std()`，`-(close-ma)/(std+1e-8)` | 价格偏离越大，反转信号越强 |
| 均值回归 | 短期反转 | `-close.pct_change(periods=N).fillna(0)` ||
| 均值回归 | 放量冲击反转 | `ret = close.pct_change()`，`vol_z=(volume-volume.rolling(N).mean())/(volume.rolling(N).std()+1e-8)`，`-(ret*vol_z).rolling(N).mean()` | 捕捉放量单边后的短期均值回归 |
| 均值回归 | VWAP偏离反转 | `typical_price=(high+low+close)/3`，`vwap=rolling_sum(typical_price*volume)/rolling_sum(volume)`，`-(close/(vwap+1e-8)-1)` | 以成交量加权均价为锚，弱化单纯均线重叠 |
| 微观结构 | 阿米胡德流动性 |`illq = abs(ret)/(volume+1e-9)`，`illq.rolling(window=N).mean()`|衡量单位成交量带来的价格变动，因子值始终为正|
| 微观结构 | 买卖压力 | `pressure = (close-low)/(high-low+1e-9)`，`pressure_ma = pressure.rolling(N).mean()`,`(pressure_ma - 0.5) * 2`| 因子值在 `[-1,1]`|
| 微观结构 | 收盘位置变化率 | `clv=((close-low)-(high-close))/(high-low+1e-9)`，`clv.diff().rolling(N).mean()` | 强调“收盘位置变化”而非静态位置 |
| 微观结构 | 上下影线不平衡 | `body_top=max(open,close)`，`body_bottom=min(open,close)`，`upper_wick=high-body_top`，`lower_wick=body_bottom-low`，`((lower_wick-upper_wick)/(high-low+1e-9)).rolling(N).mean()` | 反映盘中冲高回落/探底回升结构 |
| 动量与趋势 | 布林带突破 | `ma = close.rolling(N).mean()`，`upper = ma+2*std`，`lower = ma-2*std`,`(close - ma) / (2 * std + 1e-8)`||
| 动量与趋势 | 唐奇安通道位置 |`roll_max = high.rolling(N).max()`，`roll_min = low.rolling(N).min()`,`(close - roll_min)/(roll_max - roll_min + 1e-9) - 0.5`| 因子值在 `[-0.5,0.5]`,衡量趋势位置，买卖压力因子衡量当日强弱|
| 动量与趋势 | 双均线交叉 |`short_ma = close.rolling(fast_n).mean()`，`long_ma = close.rolling(slow_n).mean()`, `short_ma/(long_ma+1e-8) - 1`||
| 动量与趋势 | EMA价差变化 | `fast = EMA(close, N/2)`，`slow = EMA(close, N)`，`spread = fast/(slow+1e-8)-1`，`spread.diff()` | 强调趋势“加速度”，非静态趋势强度 |
| 动量与趋势 | 线性回归斜率 | `slope = rolling_regression_slope(close,N)`，`slope/(mean(abs(close))+1e-8)` | 与均线交叉不同，直接度量趋势斜率 |
| 动量与趋势 | MACD | `dif = EMA(fast_n)-EMA(slow_n)`，`dea = EMA(signal_n, dif)`,`(dif - dea) * 2` | 常用参数 `(12,26,9)`|
| 动量与趋势 | 移动平均线乖离率 | `close / close.rolling(window=N).mean() - 1` |震荡市场接近0，更偏均值回归类因子|
| 动量与趋势 | RSI 相对强弱 | `gain = (ret.where(ret > 0, 0)).rolling(window=n_value).mean()`,`loss = (ret.where(ret > 0, 0)).rolling(window=n_value).mean()`,`rs = gain / (loss + 1e-9)`,`(rsi - 50) / 100`；`rsi = 100 - 100/(1+rs)`，`rs = gain/(loss+1e-9)`，`rsi = 100 - (100 / (1 + rs))`,`(rsi - 50) / 100`| 因子值在 `[-0.5,0.5]`|
| 动量与趋势 | 时间序列动量 | `close.pct_change(periods=N).fillna(0)` |震荡市场接近0，更偏趋势类因子|
| 波动率与风险 | ATR 平均真实波幅 |`tr = max(high-low, abs(high-close.shift(1)), abs(low-close.shift(1)))`,`atr = tr.rolling(N).mean()`, `atr / close` |因子值为正 |
| 波动率与风险 | 上下行波动率倾向 |`rs = up_vol/(down_vol+1e-9)`, `vol_rsi = 100 - (100/(1+rs))`,`(vol_rsi - 50) / 100`|因子值在 `[-0.5,0.5]` |
| 波动率与风险 | 历史波动率 | `close.pct_change().rolling(window=N).std()` | 因子值为正 |
| 波动率与风险 | 日内振幅 | `((high-low)/(open+1e-9)).rolling(window=N).mean()` |因子值为正 |
| 波动率与风险 | VIXFix | `highest_close = close.rolling(N, min_periods=1).max()`,`raw = ((highest_close - low)/(highest_close + 1e-8) * 100).fillna(0)`,`raw/50 -1` |因子值在 [-1,1]，数值越大越恐慌|
| 成交量与持仓量 | CMF 蔡金资金流 | `money_flow_multiplier=((close-low)-(high-close))/(high-low+1e-8)`, `money_flow_volume = money_flow_multiplier * volume `,`rolling_sum(money_flow_volume)/ (rolling_sum(volume)+1e-8)`|衡量资金流入/出以及程度|
| 成交量与持仓量 | MFI 资金流向指标 |`typical_price = (high + low + close) / 3.0`,`raw_money_flow = typical_price * volume`, `pos_sum = raw_money_flow.where(price_change > 0, 0.0).rolling().sum` , `money_flow_ratio = pos_sum/(neg_sum+1e-8)`,`mfi = 100 - 100/(1+money_flow_ratio)`，`(mfi - 50) / 50`|因子值在[-1,1]|
| 成交量与持仓量 | OBV 能量潮 | `obv = cumsum(sign(ret) * volume)`,`obv / obv.rolling(window=N).mean() - 1`||
| 成交量与持仓量 | 持仓量变化率 | `oi / oi.shift(N) - 1`| 需要注意换月|
| 成交量与持仓量 | 价格持仓量背离 | `price_ret = close.pct_change(N)`，`oi_change = oi.pct_change(N)`,`price_ret * sign(oi_change)`（换月区间置空后前向填充） |换月处理对结果影响较大|
| 成交量与持仓量 | 价量相关性 |`daily_ret = close.pct_change()`，`vol_chg = volume.pct_change()`, `daily_ret.rolling(window=N).corr(vol_chg)` |因子绝对值越大，代表量价联动性越强|
| 成交量与持仓量 | 成交量动量 | `volume / volume.rolling(N).mean() - 1`|需要注意换月|
| 成交量与持仓量 | 价格-持仓流向 | `ret = close.pct_change()`，`oi_chg = oi.pct_change()`，`flow = sign(ret)*oi_chg`，`flow.rolling(N).mean()` | 结合方向与持仓变化，区别于单独 OI 变化率 |
