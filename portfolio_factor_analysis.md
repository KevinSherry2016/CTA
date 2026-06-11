# 因子评估结果与组合建议

- 数据源: `Evaluate/all_metrics_summary.csv`，共 5640 条记录。
- ALL 记录: 940；sector 记录: 4700。

## 1. ALL 表现较好的因子、信号与参数高原

| factor | signal | maxSharpe | plateauMean | plateauMin | robustScore | params | paramCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Volume_MFI | zscore | 1.13 | 1.09 | 1.04 | 1.07 | N40, N50 | 2 |
| TrendMomentum_DonchianChannel | zscore | 1.02 | 1.02 | 1.01 | 1.01 | N50, N60 | 2 |
| CrossSectional_TailReturnAsymmetry | zscore | 1.06 | 0.99 | 0.92 | 0.97 | N50, N40 | 2 |
| Volume_MFI | state_machine | 1.00 | 0.97 | 0.95 | 0.97 | N30, N40 | 2 |
| Volume_OiVolumeResonance | zscore | 1.03 | 0.98 | 0.93 | 0.97 | N50, N40 | 2 |
| TrendMomentum_BollingerBands | zscore | 0.96 | 0.96 | 0.95 | 0.96 | N60, N70 | 2 |
| TrendMomentum_RSI | zscore | 0.95 | 0.94 | 0.92 | 0.93 | N40, N50 | 2 |
| CrossSectional_TailReturnAsymmetry | tanh | 0.97 | 0.94 | 0.91 | 0.93 | N40, N30 | 2 |
| TrendMomentum_DualMACrossover | state_machine | 0.97 | 0.92 | 0.87 | 0.91 | F20_S40, F20_S60 | 2 |
| TrendMomentum_TimeSeriesMomentum | zscore | 0.91 | 0.90 | 0.89 | 0.90 | N40, N50 | 2 |
| TrendMomentum_LinearSlope | state_machine | 0.93 | 0.89 | 0.86 | 0.88 | N40, N50 | 2 |
| TrendMomentum_RSI | state_machine | 0.91 | 0.89 | 0.86 | 0.88 | N30, N40 | 2 |

## 2. 各 sector 表现较好的因子、信号与参数高原

### Agriculture

| factor | signal | maxSharpe | plateauMean | plateauMin | robustScore | params | paramCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Volume_OiVolumeResonance | state_machine | 0.98 | 0.89 | 0.80 | 0.86 | N40, N50 | 2 |
| Volume_OiVolumeResonance | zscore | 0.92 | 0.83 | 0.74 | 0.80 | N40, N50 | 2 |
| Volume_OiVolumeResonance | tanh | 0.89 | 0.80 | 0.71 | 0.77 | N40, N50 | 2 |
| Volume_OiVolumeResonance | raw | 0.89 | 0.80 | 0.71 | 0.77 | N40, N50 | 2 |
| CrossSectional_TailReturnAsymmetry | state_machine | 0.82 | 0.78 | 0.73 | 0.76 | N40, N50 | 2 |
| TrendMomentum_DualMACrossover | state_machine | 0.75 | 0.74 | 0.74 | 0.74 | F30_S60, F20_S60 | 2 |
| CrossSectional_TailReturnAsymmetry | tanh | 0.75 | 0.72 | 0.69 | 0.71 | N40, N50 | 2 |
| TrendMomentum_LinearSlope | state_machine | 0.74 | 0.71 | 0.68 | 0.70 | N50, N60 | 2 |

### Energy

| factor | signal | maxSharpe | plateauMean | plateauMin | robustScore | params | paramCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Volume_MFI | zscore | 1.01 | 1.00 | 0.99 | 1.00 | N30, N40 | 2 |
| TrendMomentum_DonchianChannel | zscore | 0.92 | 0.91 | 0.90 | 0.90 | N50, N40 | 2 |
| TrendMomentum_BollingerBands | zscore | 0.88 | 0.88 | 0.87 | 0.88 | N60, N50 | 2 |
| CrossSectional_TailReturnAsymmetry | zscore | 0.83 | 0.82 | 0.82 | 0.82 | N70, N50 | 2 |
| Volume_OiVolumeResonance | zscore | 0.82 | 0.80 | 0.78 | 0.80 | N50, N60 | 2 |
| TrendMomentum_RSI | zscore | 0.79 | 0.79 | 0.78 | 0.79 | N30, N50 | 2 |
| Volume_MFI | state_machine | 0.85 | 0.80 | 0.74 | 0.78 | N30, N20 | 2 |
| TrendMomentum_MovingAverageBias | zscore | 0.77 | 0.76 | 0.75 | 0.76 | N50, N60 | 2 |

### Ferrous

| factor | signal | maxSharpe | plateauMean | plateauMin | robustScore | params | paramCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MeanReversion_GapFillPressure | raw | 0.79 | 0.76 | 0.73 | 0.75 | N80, N70 | 2 |
| CrossSectional_TailReturnAsymmetry | zscore | 0.74 | 0.72 | 0.71 | 0.72 | N50, N60 | 2 |
| TrendMomentum_DonchianChannel | zscore | 0.70 | 0.70 | 0.69 | 0.69 | N60, N70 | 2 |
| TrendMomentum_BollingerBands | zscore | 0.68 | 0.68 | 0.67 | 0.68 | N80, N70 | 2 |
| TrendMomentum_MovingAverageBias | zscore | 0.66 | 0.66 | 0.66 | 0.66 | N80, N70 | 2 |
| TrendMomentum_AdaptiveBreakoutStrength | zscore | 0.70 | 0.66 | 0.62 | 0.65 | N30, N40 | 2 |
| TrendMomentum_RSI | zscore | 0.65 | 0.65 | 0.65 | 0.65 | N40, N50 | 2 |
| Volume_PriceOiDivergence | raw | 0.69 | 0.64 | 0.60 | 0.63 | N80, N60 | 2 |

### NonFerrous

| factor | signal | maxSharpe | plateauMean | plateauMin | robustScore | params | paramCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TrendMomentum_LinearSlope | state_machine | 0.95 | 0.90 | 0.86 | 0.89 | N40, N30 | 2 |
| TrendMomentum_DualMACrossover | state_machine | 1.04 | 0.92 | 0.81 | 0.89 | F20_S40, F10_S40 | 2 |
| Volume_OIPriceFlow | tanh | 0.83 | 0.83 | 0.82 | 0.83 | N70, N80 | 2 |
| Volume_OIPriceFlow | raw | 0.83 | 0.83 | 0.82 | 0.83 | N70, N80 | 2 |
| Volume_OiVolumeResonance | state_machine | 0.82 | 0.76 | 0.69 | 0.74 | N20, N30 | 2 |
| Volume_OIPriceFlow | state_machine | 0.74 | 0.73 | 0.73 | 0.73 | N20, N30 | 2 |
| TrendMomentum_DualMACrossover | zscore | 0.76 | 0.73 | 0.70 | 0.72 | F20_S40, F20_S60 | 2 |
| TrendMomentum_LinearSlope | zscore | 0.74 | 0.72 | 0.70 | 0.71 | N40, N50 | 2 |

### Precious

| factor | signal | maxSharpe | plateauMean | plateauMin | robustScore | params | paramCount |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Microstructure_WickImbalance | state_machine | 1.61 | 1.60 | 1.59 | 1.60 | N60, N70 | 2 |
| Microstructure_WickImbalance | tanh | 1.44 | 1.42 | 1.40 | 1.42 | N70, N60 | 2 |
| Microstructure_WickImbalance | raw | 1.44 | 1.42 | 1.40 | 1.41 | N70, N60 | 2 |
| MeanReversion_ExhaustionReversalComposite | state_machine | 1.37 | 1.29 | 1.22 | 1.27 | N80, N70 | 2 |
| Microstructure_BuyingSellingPressure | tanh | 1.28 | 1.27 | 1.26 | 1.26 | N70, N50 | 2 |
| Microstructure_BuyingSellingPressure | raw | 1.27 | 1.26 | 1.25 | 1.26 | N70, N50 | 2 |
| Volume_CMF | state_machine | 1.23 | 1.22 | 1.21 | 1.21 | N30, N40 | 2 |
| Volume_CMF | tanh | 1.16 | 1.15 | 1.14 | 1.15 | N70, N50 | 2 |

## 3. 四种信号处理方式整体表现

| signal | sampleCount | meanSharpe | medianSharpe | upperQuartile | maxSharpe |
| --- | --- | --- | --- | --- | --- |
| state_machine | 1410 | 0.14 | 0.20 | 0.48 | 1.61 |
| zscore | 1410 | 0.12 | 0.20 | 0.44 | 1.13 |
| tanh | 1410 | 0.15 | 0.21 | 0.44 | 1.44 |
| raw | 1410 | 0.15 | 0.20 | 0.44 | 1.44 |

## 4. 同一因子在四种信号下的表现差异（按差值排序）

| factor | raw | zscore | state_machine | tanh | bestSignal | worstSignal | spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Microstructure_WickImbalance | 0.89 | 0.26 | 0.96 | 0.89 | state_machine | zscore | 0.70 |
| MeanReversion_ExhaustionReversalComposite | 0.69 | 0.03 | 0.66 | 0.69 | tanh | zscore | 0.66 |
| MeanReversion_VolumeShockReversal | 0.35 | -0.05 | 0.47 | 0.35 | state_machine | zscore | 0.53 |
| CrossSectional_Kurtosis | 0.39 | 0.57 | 0.08 | 0.06 | zscore | tanh | 0.52 |
| Volume_OIPriceFlow | 0.89 | 0.48 | 0.84 | 0.89 | tanh | zscore | 0.42 |
| MeanReversion_GapFillPressure | 0.64 | 0.25 | 0.41 | 0.45 | raw | zscore | 0.39 |
| TrendMomentum_DualMACrossover | 0.61 | 0.76 | 0.96 | 0.61 | state_machine | raw | 0.35 |
| TrendMomentum_LinearSlope | 0.57 | 0.71 | 0.91 | 0.57 | state_machine | raw | 0.35 |
| Microstructure_BuyingSellingPressure | 0.91 | 0.58 | 0.87 | 0.92 | tanh | zscore | 0.33 |
| TrendMomentum_TimeSeriesMomentum | 0.50 | 0.73 | 0.83 | 0.54 | state_machine | raw | 0.32 |
| Volume_CMF | 0.90 | 0.58 | 0.86 | 0.91 | tanh | zscore | 0.32 |
| TrendMomentum_MovingAverageBias | 0.46 | 0.77 | 0.78 | 0.47 | state_machine | raw | 0.32 |
| MeanReversion_NoiseFader | -0.05 | 0.19 | -0.11 | -0.07 | zscore | state_machine | 0.29 |
| CrossSectional_TailReturnAsymmetry | 0.54 | 0.81 | 0.71 | 0.77 | zscore | raw | 0.28 |
| TrendMomentum_AdaptiveBreakoutStrength | 0.27 | 0.46 | 0.23 | 0.36 | zscore | state_machine | 0.24 |
| TrendMomentum_MACD | 0.43 | 0.42 | 0.59 | 0.61 | tanh | zscore | 0.19 |
| CrossSectional_Skewness | 0.55 | 0.58 | 0.39 | 0.48 | zscore | state_machine | 0.19 |
| MeanReversion_ShortTermReversal | -0.03 | -0.21 | -0.11 | -0.04 | raw | zscore | 0.18 |

## 5. 组合候选（用于下一步回测验证）

### 5.1 ALL 口径候选（按单腿 Sharpe 排序）

| factor | sector | param | signal | sharpeRatio | strategyFile |
| --- | --- | --- | --- | --- | --- |
| Volume_MFI | all | N40 | zscore | 1.13 | Volume_MFI_ALL_N40_ZSCORE_Position.csv |
| CrossSectional_TailReturnAsymmetry | all | N50 | zscore | 1.06 | CrossSectional_TailReturnAsymmetry_ALL_N50_ZSCORE_Position.csv |
| Volume_OiVolumeResonance | all | N50 | zscore | 1.03 | Volume_OiVolumeResonance_ALL_N50_ZSCORE_Position.csv |
| TrendMomentum_DonchianChannel | all | N50 | zscore | 1.02 | TrendMomentum_DonchianChannel_ALL_N50_ZSCORE_Position.csv |
| TrendMomentum_DualMACrossover | all | F20_S40 | state_machine | 0.97 | TrendMomentum_DualMACrossover_ALL_F20_S40_STATE_MACHINE_Position.csv |
| TrendMomentum_BollingerBands | all | N60 | zscore | 0.96 | TrendMomentum_BollingerBands_ALL_N60_ZSCORE_Position.csv |
| TrendMomentum_RSI | all | N40 | zscore | 0.95 | TrendMomentum_RSI_ALL_N40_ZSCORE_Position.csv |
| TrendMomentum_LinearSlope | all | N40 | state_machine | 0.93 | TrendMomentum_LinearSlope_ALL_N40_STATE_MACHINE_Position.csv |

### 5.2 Sector 分开口径候选（按单腿 Sharpe 排序）

| factor | sector | param | signal | sharpeRatio | strategyFile |
| --- | --- | --- | --- | --- | --- |
| Microstructure_WickImbalance | Precious | N60 | state_machine | 1.61 | Microstructure_WickImbalance_Precious_N60_STATE_MACHINE_Position.csv |
| MeanReversion_ExhaustionReversalComposite | Precious | N80 | state_machine | 1.37 | MeanReversion_ExhaustionReversalComposite_Precious_N80_STATE_MACHINE_Position.csv |
| Microstructure_BuyingSellingPressure | Precious | N70 | tanh | 1.28 | Microstructure_BuyingSellingPressure_Precious_N70_TANH_Position.csv |
| Volume_CMF | Precious | N30 | state_machine | 1.23 | Volume_CMF_Precious_N30_STATE_MACHINE_Position.csv |
| TrendMomentum_DualMACrossover | Precious | F20_S40 | state_machine | 1.17 | TrendMomentum_DualMACrossover_Precious_F20_S40_STATE_MACHINE_Position.csv |
| TrendMomentum_LinearSlope | Precious | N40 | state_machine | 1.09 | TrendMomentum_LinearSlope_Precious_N40_STATE_MACHINE_Position.csv |
| Volume_OIPriceFlow | Precious | N20 | tanh | 1.07 | Volume_OIPriceFlow_Precious_N20_TANH_Position.csv |
| Volume_OiVolumeResonance | Precious | N30 | state_machine | 1.06 | Volume_OiVolumeResonance_Precious_N30_STATE_MACHINE_Position.csv |
| TrendMomentum_TimeSeriesMomentum | Precious | N30 | state_machine | 1.05 | TrendMomentum_TimeSeriesMomentum_Precious_N30_STATE_MACHINE_Position.csv |
| TrendMomentum_RSI | Precious | N30 | state_machine | 1.04 | TrendMomentum_RSI_Precious_N30_STATE_MACHINE_Position.csv |

## 结论

1. 删除四个因子后，整体样本从 6408 降至 5640，分析结果已同步更新。
2. 建议优先从参数高原（而非单点峰值）挑选策略，避免参数尖峰误导。
3. 信号方式上建议保留 zscore/state_machine 与 raw/tanh 的并行对照，但按因子分别筛选，不做全局一刀切。
4. 第 5 节仅给出候选腿，最终组合仍应基于日度 PnL 相关性和回撤约束再筛选。

## 6. 各 Sector 品种：趋势/反转适配与较差品种

口径说明：
- 趋势适配：`TrendMomentum_*` 中该 sector Sharpe 前3策略的品种平均Sharpe。
- 反转适配：`MeanReversion_*` 中该 sector Sharpe 前3策略的品种平均Sharpe。
- 表现较差：优先取趋势和反转都<=0 的品种；若不存在，则给出“相对较弱”品种（该 sector 综合Sharpe尾部）。

### Agriculture
- 趋势参考策略: TrendMomentum_TimeSeriesMomentum_Agriculture_N40_STATE_MACHINE_Position.csv, TrendMomentum_RSI_Agriculture_N40_STATE_MACHINE_Position.csv, TrendMomentum_DualMACrossover_Agriculture_F30_S60_STATE_MACHINE_Position.csv
- 反转参考策略: MeanReversion_ExhaustionReversalComposite_Agriculture_N70_RAW_Position.csv, MeanReversion_ExhaustionReversalComposite_Agriculture_N80_RAW_Position.csv, MeanReversion_ExhaustionReversalComposite_Agriculture_N70_TANH_Position.csv

**更适合趋势的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| JD.DCE | 0.82 | -0.25 | 0.29 |
| LH.DCE | 0.78 | -0.03 | 0.37 |
| CS.DCE | 0.35 | -0.52 | -0.08 |
| C.DCE | 0.25 | 0.02 | 0.13 |
| UR.ZCE | 0.22 | 0.07 | 0.14 |
| B.DCE | 0.21 | -0.02 | 0.09 |

**更适合反转的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| AP.ZCE | 0.39 | 1.18 | 0.78 |
| OI.ZCE | 0.37 | 0.73 | 0.55 |

**表现较差的品种（绝对）**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| A.DCE | -0.18 | -0.10 | -0.14 |
| RM.ZCE | -0.19 | -0.04 | -0.11 |

### Energy
- 趋势参考策略: TrendMomentum_DonchianChannel_Energy_N50_ZSCORE_Position.csv, TrendMomentum_DonchianChannel_Energy_N40_ZSCORE_Position.csv, TrendMomentum_BollingerBands_Energy_N60_ZSCORE_Position.csv
- 反转参考策略: MeanReversion_VolumeShockReversal_Energy_N80_STATE_MACHINE_Position.csv, MeanReversion_VolumeShockReversal_Energy_N30_STATE_MACHINE_Position.csv, MeanReversion_VolumeShockReversal_Energy_N50_STATE_MACHINE_Position.csv

**更适合趋势的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| SC.INE | 1.09 | 0.18 | 0.63 |
| SA.ZCE | 0.97 | 0.13 | 0.55 |
| BZ.DCE | 0.90 | -0.40 | 0.25 |
| PL.ZCE | 0.77 | 0.18 | 0.48 |
| FG.ZCE | 0.72 | 0.56 | 0.64 |
| TA.ZCE | 0.60 | 0.49 | 0.55 |

**更适合反转的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| FU.SHF | 0.57 | 0.68 | 0.62 |
| L.DCE | 0.20 | 0.66 | 0.43 |
| PF.ZCE | 0.02 | 0.63 | 0.32 |
| SH.ZCE | 0.11 | 0.41 | 0.26 |
| PP.DCE | 0.00 | 0.29 | 0.15 |
| PG.DCE | -0.12 | 0.26 | 0.07 |

**表现相对较弱的品种（尾部）**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| RU.SHF | 0.04 | -0.18 | -0.07 |
| PX.ZCE | -0.14 | 0.07 | -0.04 |
| PG.DCE | -0.12 | 0.26 | 0.07 |
| LU.INE | 0.17 | 0.09 | 0.13 |
| PP.DCE | 0.00 | 0.29 | 0.15 |
| EB.DCE | 0.59 | -0.18 | 0.20 |

### Ferrous
- 趋势参考策略: TrendMomentum_DonchianChannel_Ferrous_N60_ZSCORE_Position.csv, TrendMomentum_AdaptiveBreakoutStrength_Ferrous_N30_ZSCORE_Position.csv, TrendMomentum_DonchianChannel_Ferrous_N70_ZSCORE_Position.csv
- 反转参考策略: MeanReversion_GapFillPressure_Ferrous_N80_RAW_Position.csv, MeanReversion_GapFillPressure_Ferrous_N70_RAW_Position.csv, MeanReversion_GapFillPressure_Ferrous_N10_RAW_Position.csv

**更适合趋势的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| HC.SHF | 0.92 | 0.13 | 0.52 |
| SM.ZCE | 0.55 | 0.45 | 0.50 |
| RB.SHF | 0.45 | -0.17 | 0.14 |
| JM.DCE | 0.35 | -0.41 | -0.03 |
| J.DCE | 0.31 | -0.26 | 0.03 |
| SS.SHF | 0.24 | -0.20 | 0.02 |

**更适合反转的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| I.DCE | 0.36 | 0.70 | 0.53 |

**表现相对较弱的品种（尾部）**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| JM.DCE | 0.35 | -0.41 | -0.03 |
| SS.SHF | 0.24 | -0.20 | 0.02 |
| SF.ZCE | 0.12 | -0.08 | 0.02 |
| J.DCE | 0.31 | -0.26 | 0.03 |
| RB.SHF | 0.45 | -0.17 | 0.14 |
| SM.ZCE | 0.55 | 0.45 | 0.50 |

### NonFerrous
- 趋势参考策略: TrendMomentum_DualMACrossover_NonFerrous_F20_S40_STATE_MACHINE_Position.csv, TrendMomentum_LinearSlope_NonFerrous_N40_STATE_MACHINE_Position.csv, TrendMomentum_LinearSlope_NonFerrous_N30_STATE_MACHINE_Position.csv
- 反转参考策略: MeanReversion_GapFillPressure_NonFerrous_N60_RAW_Position.csv, MeanReversion_GapFillPressure_NonFerrous_N50_RAW_Position.csv, MeanReversion_GapFillPressure_NonFerrous_N80_RAW_Position.csv

**更适合趋势的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| SN.SHF | 1.02 | 0.31 | 0.66 |
| AL.SHF | 0.68 | -0.08 | 0.30 |
| AO.SHF | 0.57 | -0.35 | 0.11 |
| AD.SHF | 0.30 | -0.12 | 0.09 |

**更适合反转的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| CU.SHF | 0.50 | 0.82 | 0.66 |
| PB.SHF | -0.13 | 0.39 | 0.13 |

**表现相对较弱的品种（尾部）**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| AD.SHF | 0.30 | -0.12 | 0.09 |
| AO.SHF | 0.57 | -0.35 | 0.11 |
| PB.SHF | -0.13 | 0.39 | 0.13 |
| AL.SHF | 0.68 | -0.08 | 0.30 |
| ZN.SHF | 0.33 | 0.42 | 0.37 |
| NI.SHF | 0.57 | 0.66 | 0.62 |

### Precious
- 趋势参考策略: TrendMomentum_DualMACrossover_Precious_F20_S40_STATE_MACHINE_Position.csv, TrendMomentum_LinearSlope_Precious_N40_STATE_MACHINE_Position.csv, TrendMomentum_TimeSeriesMomentum_Precious_N30_STATE_MACHINE_Position.csv
- 反转参考策略: MeanReversion_ExhaustionReversalComposite_Precious_N80_STATE_MACHINE_Position.csv, MeanReversion_ExhaustionReversalComposite_Precious_N70_STATE_MACHINE_Position.csv, MeanReversion_ExhaustionReversalComposite_Precious_N80_TANH_Position.csv

**更适合趋势的品种**

无明显候选

**更适合反转的品种**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| AU.SHF | 1.01 | 1.18 | 1.09 |

**表现相对较弱的品种（尾部）**

| symbol | trendSharpe | reversalSharpe | overallSharpe |
| --- | --- | --- | --- |
| AG.SHF | 0.87 | 0.85 | 0.86 |
| AU.SHF | 1.01 | 1.18 | 1.09 |


## Symbol Sharpe Ranking (L2)

### Trend
| symbol | sharpeRatio |
|---|---:|
| SC.INE | 1.098027 |
| AU.SHF | 1.011368 |
| SA.ZCE | 0.953104 |
| SN.SHF | 0.947188 |
| HC.SHF | 0.937270 |
| BZ.DCE | 0.932740 |
| JD.DCE | 0.874129 |
| AG.SHF | 0.852855 |
| PL.ZCE | 0.844411 |
| LH.DCE | 0.815512 |
| FG.ZCE | 0.780804 |
| NI.SHF | 0.652334 |
| FU.SHF | 0.623982 |
| CF.ZCE | 0.623455 |
| EB.DCE | 0.621103 |
| TA.ZCE | 0.606096 |
| V.DCE | 0.586336 |
| SM.ZCE | 0.556431 |
| SP.SHF | 0.518500 |
| CU.SHF | 0.497741 |
| BU.SHF | 0.482397 |
| SF.ZCE | 0.474837 |
| AO.SHF | 0.471636 |
| AL.SHF | 0.445948 |
| RB.SHF | 0.432698 |
| AP.ZCE | 0.417051 |
| AD.SHF | 0.416660 |
| P.DCE | 0.409823 |
| MA.ZCE | 0.400254 |
| OI.ZCE | 0.386474 |
| I.DCE | 0.370476 |
| JM.DCE | 0.347315 |
| J.DCE | 0.341742 |
| ZN.SHF | 0.330981 |
| CS.DCE | 0.319024 |
| M.DCE | 0.265136 |
| B.DCE | 0.223647 |
| UR.ZCE | 0.198513 |
| L.DCE | 0.195579 |
| C.DCE | 0.188465 |
| LU.INE | 0.184847 |
| EG.DCE | 0.170185 |
| SS.SHF | 0.143638 |
| SH.ZCE | 0.132538 |
| PF.ZCE | 0.128061 |
| Y.DCE | 0.102076 |
| SR.ZCE | 0.074128 |
| RU.SHF | 0.008442 |
| PP.DCE | -0.069491 |
| RM.ZCE | -0.156416 |
| PX.ZCE | -0.157762 |
| PB.SHF | -0.212059 |
| PG.DCE | -0.245491 |
| A.DCE | -0.271187 |

### Reversion
| symbol | sharpeRatio |
|---|---:|
| AU.SHF | 1.384124 |
| AP.ZCE | 1.182041 |
| I.DCE | 0.842343 |
| CU.SHF | 0.815183 |
| L.DCE | 0.757884 |
| TA.ZCE | 0.737181 |
| OI.ZCE | 0.732472 |
| PF.ZCE | 0.723770 |
| P.DCE | 0.723569 |
| FU.SHF | 0.665736 |
| AG.SHF | 0.643578 |
| FG.ZCE | 0.613699 |
| NI.SHF | 0.609392 |
| SH.ZCE | 0.510503 |
| SM.ZCE | 0.444515 |
| ZN.SHF | 0.406830 |
| PB.SHF | 0.382299 |
| PP.DCE | 0.380441 |
| SN.SHF | 0.321932 |
| V.DCE | 0.320764 |
| PG.DCE | 0.293459 |
| BU.SHF | 0.273290 |
| Y.DCE | 0.271890 |
| CF.ZCE | 0.262179 |
| PX.ZCE | 0.247017 |
| LU.INE | 0.238727 |
| SC.INE | 0.216741 |
| SP.SHF | 0.203835 |
| EG.DCE | 0.178648 |
| PL.ZCE | 0.152486 |
| HC.SHF | 0.129268 |
| SA.ZCE | 0.074752 |
| M.DCE | 0.068135 |
| C.DCE | 0.062790 |
| SF.ZCE | 0.054082 |
| MA.ZCE | 0.051630 |
| B.DCE | 0.025578 |
| LH.DCE | -0.007443 |
| RM.ZCE | -0.021961 |
| UR.ZCE | -0.080833 |
| AL.SHF | -0.082156 |
| A.DCE | -0.114829 |
| AD.SHF | -0.133787 |
| RB.SHF | -0.174499 |
| SR.ZCE | -0.181259 |
| SS.SHF | -0.196797 |
| RU.SHF | -0.242462 |
| J.DCE | -0.258761 |
| JD.DCE | -0.262277 |
| EB.DCE | -0.347603 |
| JM.DCE | -0.406850 |
| AO.SHF | -0.407378 |
| BZ.DCE | -0.423364 |
| CS.DCE | -0.451592 |

### Alternative
| symbol | sharpeRatio |
|---|---:|
| AU.SHF | 1.347189 |
| AG.SHF | 1.345778 |
| SC.INE | 1.130748 |
| EB.DCE | 0.928523 |
| FG.ZCE | 0.863954 |
| SN.SHF | 0.808327 |
| SA.ZCE | 0.802241 |
| AO.SHF | 0.755539 |
| RB.SHF | 0.753595 |
| TA.ZCE | 0.749298 |
| BZ.DCE | 0.736942 |
| AP.ZCE | 0.694705 |
| HC.SHF | 0.689333 |
| SP.SHF | 0.685694 |
| M.DCE | 0.628490 |
| LH.DCE | 0.628319 |
| I.DCE | 0.618971 |
| SM.ZCE | 0.614051 |
| PL.ZCE | 0.611972 |
| JD.DCE | 0.611436 |
| EG.DCE | 0.581840 |
| L.DCE | 0.548208 |
| AL.SHF | 0.503578 |
| CF.ZCE | 0.449963 |
| MA.ZCE | 0.448602 |
| V.DCE | 0.445552 |
| JM.DCE | 0.432769 |
| P.DCE | 0.423211 |
| C.DCE | 0.417157 |
| UR.ZCE | 0.410227 |
| NI.SHF | 0.410180 |
| BU.SHF | 0.396443 |
| PX.ZCE | 0.386679 |
| ZN.SHF | 0.356226 |
| SS.SHF | 0.342331 |
| LU.INE | 0.339421 |
| PP.DCE | 0.336437 |
| CU.SHF | 0.302799 |
| PB.SHF | 0.248402 |
| AD.SHF | 0.234864 |
| SH.ZCE | 0.209057 |
| OI.ZCE | 0.208297 |
| B.DCE | 0.149029 |
| CS.DCE | 0.146263 |
| Y.DCE | 0.125206 |
| RU.SHF | 0.116686 |
| A.DCE | 0.027345 |
| SF.ZCE | 0.021528 |
| J.DCE | -0.058039 |
| PG.DCE | -0.068955 |
| RM.ZCE | -0.092693 |
| SR.ZCE | -0.124917 |
| PF.ZCE | -0.172985 |
| FU.SHF | -0.230137 |


Sharpe: 2.2504
年化收益: 26.4892
年化波动: 11.7710
累计PnL: 216.0460
最大回撤: -9.0043
