# 因子批量测试与评估计(Master Plan)

## 1. 核心目标
系统性地开发、回测和评估多个量化因子（宏观、微观、量价等），并分析它们与基础趋势策略之间的相关性。最终目标是遍历所有因子进行测试，找出具有正向收益预期且相互之间低相关性的组合，而不是固定某一基础趋势框架。

## 2. 待测因子(Factor Pool)
在进行自动化测试时，将按照下表的因子清单逐一推进。

| 因子名称 (Factor Name) | 因子类别 | 逻辑简介| 测试状态| 评估与相关性计算|
| :--- | :--- | :--- | :---: | :---: |
| `CrossSectional_Kurtosis` | CrossSectional | 截面峰度 | [x] | [x] |
| `CrossSectional_OvernightVsIntraday` | CrossSectional | 隔夜与日内收益率差| [x] | [x] |
| `CrossSectional_Skewness` | CrossSectional | 截面偏度 | [x] | [x] |
| `Microstructure_AmihudIlliquidity` | Microstructure | Amihud缺乏流动性指标| [x] | [x] |
| `Microstructure_BuyingSellingPressure` | Microstructure | 收盘价在日内高低点的位置 | [x] | [x] |
| `TrendMomentum_BollingerBands` | TrendMomentum | 布林带突破| [x] | [x] |
| `TrendMomentum_DonchianChannel` | TrendMomentum | 唐奇安通道突破 | [x] | [x] |
| `TrendMomentum_DualMACrossover` | TrendMomentum | 双均线交破| [x] | [x] |
| `TrendMomentum_MACD` | TrendMomentum | MACD动量 | [x] | [x] |
| `TrendMomentum_MovingAverageBias` | TrendMomentum | 背离度| [x] | [x] |
| `TrendMomentum_RSI` | TrendMomentum | 相对强弱指数 | [x] | [x] |
| `TrendMomentum_TimeSeriesMomentum` | TrendMomentum | 时序动量 | [x] | [x] |
| `Volatility_ATR` | Volatility | 真实波动幅度 | [x] | [x] |
| `Volatility_DownsideUpsideVolatility` | Volatility | 上下行波动率倾向 | [x] | [x] |
| `Volatility_HistoricalVolatility` | Volatility | 历史收益率波动率 | [x] | [x] |
| `Volatility_IntradayAmplitude` | Volatility | 日内振幅 | [x] | [x] |
| `Volume_OBV` | Volume | 能量潮指| [x] | [x] |
| `Volume_OpenInterestROC` | Volume | 持仓量变化率 (OI) | [x] | [x] |
| `Volume_PriceVolumeCorrelation` | Volume | 量价相关性| [x] | [x] |
| `Volume_VolumeMomentum` | Volume | 成交量长短期放缩| [x] | [x] |

## 3. 执行工作(Workflow)
对于每一个新进入测试管线的因子，执行以下自动化流
1. **[Step 1] 开发策略**：按照 `02_Implementation_SOP.md` 编写代码
2. **[Step 2] 运行与回测**：自动运行生成的 `Strategy.py`，生成一个`_BacktestResult.csv`，

## 4. 统一汇总
所有因子都运行完成后，按照`03_Evaluation_Protocol.md`执行以下自动化操作
1. **[Step 1] 汇总表格**：寻找整体参数高原并检验年化Sharpe 一致性，综合评估过拟合（偏移）风险，挑选出最稳健参数并登记其 Sharpe 和 POT
2. **[Step 2] 计算相关性**：提取每日盈亏（PnL）数据，计算 PnL 相关性矩阵
3. **[Step 3] 总结输出**：归档测试报告，并更新本表状态
