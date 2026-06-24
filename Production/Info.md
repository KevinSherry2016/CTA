### Trend (趋势类)
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

### Carry (期限结构类)
| 因子 | 板块 | 参数 | 信号 |
| --- | --- | --- | --- |
| Carry_Momentum | Agriculture | N10 | state_machine |
| Carry_Momentum | Energy | N10 | state_machine |
| Carry_Momentum | Energy | N20 | state_machine |
| Carry_Momentum | Energy | N30 | state_machine |
| Carry_Momentum | Ferrous | N60 | state_machine |
| Carry_Momentum | NonFerrous | N10 | tanh |
| Carry_Momentum | NonFerrous | N60 | tanh |
| Carry_Momentum | NonFerrous | N80 | tanh |
| Carry_Momentum | Precious | N80 | state_machine |

**L2 绩效**: Sharpe=1.6276, 累计PnL=212.61, 年化收益≈25.9%, 年化波动≈15.9%

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

### 四类策略组合
**等波动率合并（Trend+Carry+Reversion+Alternative）**: 输出组合为 L2_EqualVol_Trend_Carry_Reversion_Alternative
**说明**: 组合逻辑见 Production/PortfolioEqualVolBlend.py，权重按等波动率计算并对最终组合做标准化
