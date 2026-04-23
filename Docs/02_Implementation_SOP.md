# 因子代码开发标准作业程序 (Implementation SOP)

## 1. 命名规范与目录结构
所有因子的回测脚本全部独立开发，脚本需放置于 `Strategy/` 目录下：
- **独立策略模型 (Standalone Strategy)**: 命名必须完全复制 `Factor/` 目录下的因子文件名，即 `{Category}_{FactorName}.py`（例如 `Strategy/CrossSectional_Kurtosis.py`）。

## 2. 因子策略模型开发规范 (`{Category}_{FactorName}.py`)
1. **统一因子计算逻辑 (Factor Calculation)**：每个因子计算时，**都必须直接使用并完全保持与 `Factor/` 目录下同名因子的原始计算逻辑一致。
2. **参数寻优范围 (Parameter Expansion)：
   - **单参数因子** 参数N的范围设定在`[10, 20, 30, 40, 50, 60]` 测试空间。
   - **多参数因子**（DualMACrossover和MACD）：直接使用因子中的 `param_list` 进行回测。
3. **因子保留双模式 (Dual Modes)**：回测流程必须同时涵盖这两种模式（'RAW'和'STATE_MACHINE'），确保一键运行即可同时计算两种逻辑：
   - `RAW` 模式：直接保留能够反映截距和偏移度的相对连续值（如 `(fast-slow)/slow`）。
   - `STATE_MACHINE` 模式：把因子强行粗粒化，通常用 `np.sign(raw_sig)` 映射到 `[-1, 0, 1]` 的看多/看空标志位。最终仓位乘数变量应统称为 `states`。
4. **回测板块**：仅测试NonFerrous板块。
5. **波动率缩放**：目标仓位必须使用 `states / vol` （20日倒数标准差）处理。
6. **仓位平滑**：利用 `POSITION_SMOOTH_DAYS = 10` 的 EMA 进行仓位平滑，减少剧烈换手。
7. **输出结果**：每个因子输出 1 个合并的回测结果文件：`{Category}_{FactorName}_BacktestResult.csv`。包括了因子名称，参数，因子模式（raw和state machine），sharpRatio和pot。
每个因子同时也输出不同参数情况下的仓位文件：`{Category}_{FactorName}_{FactorMode}_Position.csv`。
