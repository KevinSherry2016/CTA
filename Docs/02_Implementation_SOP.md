# 因子代码开发标准作业程序 (Implementation SOP)

## 1. 命名规范与目录结构
所有因子的回测脚本全部独立开发，脚本需放置于 `Strategy/` 目录下：
- **独立策略模型 (Standalone Strategy)**: 命名必须完全复制 `Factor/` 目录下的因子文件名，即 `{Category}_{FactorName}.py`（例如 `Strategy/CrossSectional_Kurtosis.py`）。不再区分开发过滤挂件。

## 2. 因子策略模型开发规范 (`{Category}_{FactorName}.py`)
1. **统一因子计算逻辑 (Factor Calculation Sync)**：每个因子计算时，**都必须直接使用并完全保持与 `Factor/` 目录下同名因子的原始计算逻辑一致**，以该因子文件输出的 `raw_sig` 为准。导入全版所有有效品种数据。
2. **扩展参数寻优范围 (Parameter Expansion)**：基于 `Factor/` 计算出的指标原始值 `raw_sig` 后需要进行平滑或寻优时，**在保持参数适当密集的同时扩大参数的选取范围**，以利于更好地观察并寻找真正的参数高原。例如：将原有的 `[10, 20, 30]` 扩展到 `[10, 20, 30, 40, 50, 60]` 这类更宽广的测试空间。
   - **多参数情况**（如 MACD 的 `fast_n, slow_n, signal_n` 或 DualMA 的 `fast_n, slow_n`）：直接使用因子中的 `param_list` 即可。
3. **因子保留双模式遍历 (Dual Modes)**：回测流程必须同时涵盖这两种模式（例如通过外层循环 `for FACTOR_MODE in ['RAW', 'STATE_MACHINE']:`），确保一键运行即可同时计算两种逻辑：
   - `RAW` 模式：直接保留能够反映截距和偏移度的相对连续值（如 `(fast-slow)/slow`）。
   - `STATE_MACHINE` 模式：把因子强行粗粒化，通常用 `np.sign(raw_sig)` 映射到 `[-1, 0, 1]` 的看多/看空标志位。最终仓位乘数变量应统称为 `states`。
4. **波动率缩放**：目标仓位必须使用 `states / vol` （20日倒数标准差）处理。
5. **仓位平滑**：利用 `POSITION_SMOOTH_DAYS = 10` 的 EMA 进行仓位平滑，减少剧烈换手。
6. **输出结果**：每个因子不论测试多少组参数，最终只输出 2 个合并的回测结果文件：`{Category}_{FactorName}_RAW_Ferrous_BacktestResult.csv` 与 `{Category}_{FactorName}_STATE_MACHINE_Ferrous_BacktestResult.csv`（及对应的 Position 文件），文件内部通过 `param` 或 `N` 等字段区分不同的参数组合。绝不能为每组参数单独生成一个文件。