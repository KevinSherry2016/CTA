Version1
| 因子 | 回测品种 | 参数 | 信号 |
| --- | --- | --- | --- |
| TrendMomentum_MovingAverageBias | Ferrous | N30 | state_machine |
| CrossSectional_OvernightVsIntraday | Ferrous | N20 | state_machine |
| TrendMomentum_MovingAverageBias | NonFerrous_CuAl | N40 | state_machine |
| TrendMomentum_MACD | NonFerrous_CuAl | fast_n=24, slow_n=52, signal_n=18 | state_machine |
| TrendMomentum_DualMACrossover | NonFerrous_Others | fast_n=20, slow_n=60 | state_machine |
| Microstructure_BuyingSellingPressure | NonFerrous_Others | N30 | state_machine |
| TrendMomentum_MovingAverageBias | Energy | N40 | state_machine |
| Volume_CMF | Energy | N40 | raw |
| TrendMomentum_DualMACrossover | Precious | fast_n=20, slow_n=40 | state_machine |
| TrendMomentum_DualMACrossover | Precious | fast_n=20, slow_n=60 | state_machine |
| Microstructure_BuyingSellingPressure | Precious | N40 | state_machine |
| Volume_CMF | Agriculture_Oils | N50 | state_machine |
| TrendMomentum_DonchianChannel | Agriculture_Softs | N50 | state_machine |
说明：
Ferrous中不包括SF SM
Agriculture仅包括油脂油料和软商品



Trend
| 因子 | 板块 | 参数 | 信号 |
| --- | --- | --- | --- |
| TrendMomentum_TimeSeriesMomentum | Agriculture | N40 | state_machine |
| TrendMomentum_LinearSlope | Agriculture | N50 | state_machine |
| TrendMomentum_DonchianChannel | Energy | N25 | zscore |
| TrendMomentum_DonchianChannel | Energy | N75 | zscore |
| TrendMomentum_AdaptiveBreakoutStrength | Ferrous | N30 | zscore |
| TrendMomentum_DonchianChannel | Ferrous | N60 | zscore |
| TrendMomentum_LinearSlope | NonFerrous | N20 | state_machine |
| TrendMomentum_LinearSlope | NonFerrous | N70 | state_machine |
| TrendMomentum_TimeSeriesMomentum | Precious | N30 | state_machine |
| TrendMomentum_LinearSlope | Precious | N40 | state_machine |

**L2 绩效**: Sharpe=1.3486, 累计PnL=162.01, 年化收益≈15.5%, 年化波动≈11.5%

### Reversion (反转类)
| 因子 | 板块 | 参数 | 信号 |
| --- | --- | --- | --- |
| MeanReversion_GapFillPressure | Agriculture | N30 | tanh |
| MeanReversion_ExhaustionReversalComposite | Agriculture | N70 | tanh |
| MeanReversion_VolumeShockReversal | Energy | N30 | state_machine |
| MeanReversion_VolumeShockReversal | Energy | N80 | state_machine |
| MeanReversion_GapFillPressure | Ferrous | N10 | state_machine |
| MeanReversion_GapFillPressure | Ferrous | N80 | state_machine |
| MeanReversion_GapFillPressure | NonFerrous | N20 | state_machine |
| MeanReversion_GapFillPressure | NonFerrous | N80 | state_machine |
| MeanReversion_ExhaustionReversalComposite | Precious | N10 | tanh |
| MeanReversion_ExhaustionReversalComposite | Precious | N80 | state_machine |

**L2 绩效**: Sharpe=1.1593, 累计PnL=143.32, 年化收益≈13.8%, 年化波动≈11.9%


### Alternative (其他类)
| 因子 | 板块 | 参数 | 信号 |
| --- | --- | --- | --- |
| Volume_OiVolumeResonance | Agriculture | N40 | state_machine |
| CrossSectional_TailReturnAsymmetry | Agriculture | N70 | tanh |
| Volume_MFI | Energy | N30 | zscore |
| CrossSectional_TailReturnAsymmetry | Energy | N70 | zscore |
| Volume_PriceVolumeCorrelation | Ferrous | N40 | zscore |
| CrossSectional_Skewness | Ferrous | N80 | zscore |
| Volume_OiVolumeResonance | NonFerrous | N20 | state_machine |
| Volume_OIPriceFlow | NonFerrous | N70 | tanh |
| Microstructure_WickImbalance | Precious | N40 | state_machine |
| Microstructure_WickImbalance | Precious | N60 | state_machine |

**L2 绩效**: Sharpe=1.5746, 累计PnL=208.88, 年化收益≈20.2%, 年化波动≈12.9%

### 三类策略组合
**等波动率合并**: Sharpe=1.9126, 累计PnL=170.45, 权重=(0.3476, 0.3377, 0.3147)
**Trend+Reversion等波动率**: Sharpe=1.7241, 累计PnL=155.90, 权重=(0.673, 0.327)
**Trend+Reversion 1:1**: Sharpe=1.7706, 累计PnL=152.67
**三类相关性**: T-R=0.0007, T-A=0.8012, R-A=-0.0239
