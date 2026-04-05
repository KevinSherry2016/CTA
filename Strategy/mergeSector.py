from pathlib import Path

import pandas as pd


RESULT_DIR = Path('./Result')

# 若指定 INPUT_FILES，则优先使用该列表；否则使用 INPUT_GLOB 自动收集。
INPUT_FILES = [
    './Result/MovingAverageV4_2_StockIndex_slow_ma_slope_over_atr_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
    './Result/MovingAverageV4_2_Bond_slow_ma_slope_over_atr_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
    './Result/MovingAverageV4_2_Energy_fast_slow_gap_over_atr_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
    './Result/MovingAverageV4_2_Ferrous_price_minus_slow_ma_over_atr_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
    './Result/MovingAverageV4_2_NonFerrous_fast_slow_gap_over_vol_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
    './Result/MovingAverageV4_2_Precious_fast_slow_gap_over_vol_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
    './Result/MovingAverageV4_2_Agriculture_slow_ma_slope_over_atr_F_8_S_40_ATR_15_SLP_5_ZO_0.6_ZC_0.2_T_5_trend_position_normalized.csv',
]   
INPUT_GLOB = '*_position_normalized.csv'

# 可选权重。未出现在该字典中的文件默认权重为 1.0。
# key 使用文件名（不含路径）更稳妥。
FILE_WEIGHTS = {
    # 'xxx_position_normalized.csv': 1.0,
}

# 合并模式：
# sum     -> 直接加总各标准化仓位
# average -> 按权重求平均，避免总仓位规模随着文件数线性放大
MERGE_MODE = 'sum'

# 平滑窗口（天数）。0 或 1 表示不平滑。
SMOOTH_WINDOW = 1

OUTPUT_PATH = RESULT_DIR / 'merged_sector_position.csv'


def collect_input_files() -> list[Path]:
    """收集待合并的标准化仓位文件。"""
    if INPUT_FILES:
        return [Path(path) for path in INPUT_FILES]

    files = sorted(RESULT_DIR.glob(INPUT_GLOB))
    files = [path for path in files if path.name != OUTPUT_PATH.name]
    return files


def load_position_csv(path: Path) -> pd.DataFrame:
    """读取单个标准化仓位 CSV。"""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df.astype(float)


def merge_positions(position_dfs: list[tuple[Path, pd.DataFrame]]) -> pd.DataFrame:
    """对齐日期与品种后合并多个仓位表。"""
    all_index = sorted({idx for _, df in position_dfs for idx in df.index})
    all_columns = sorted({col for _, df in position_dfs for col in df.columns})

    merged = pd.DataFrame(0.0, index=all_index, columns=all_columns, dtype='float64')
    total_weight = 0.0

    for path, df in position_dfs:
        weight = float(FILE_WEIGHTS.get(path.name, 1.0))
        aligned = df.reindex(index=all_index, columns=all_columns).fillna(0.0)
        merged = merged + aligned * weight
        total_weight += weight
        print(f'已合并: {path.name}  weight={weight}')

    if MERGE_MODE == 'average':
        if total_weight == 0:
            raise ValueError('总权重为 0，无法做 average 合并。')
        merged = merged / total_weight
    elif MERGE_MODE != 'sum':
        raise ValueError("MERGE_MODE must be either 'sum' or 'average'.")

    return merged.sort_index()


def main():
    input_files = collect_input_files()
    if not input_files:
        raise ValueError('未找到可合并的标准化仓位文件，请检查 INPUT_FILES 或 INPUT_GLOB。')

    missing_files = [path for path in input_files if not path.exists()]
    if missing_files:
        missing_text = '\n'.join(str(path) for path in missing_files)
        raise FileNotFoundError(f'以下文件不存在:\n{missing_text}')

    print(f'待合并文件数量: {len(input_files)}')
    position_dfs = [(path, load_position_csv(path)) for path in input_files]
    merged = merge_positions(position_dfs)

    if SMOOTH_WINDOW > 1:
        merged = merged.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
        print(f'平滑窗口: {SMOOTH_WINDOW} 天')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, encoding='utf-8-sig')

    print(f'合并完成: {OUTPUT_PATH}')
    print(f'日期数: {len(merged.index)}')
    print(f'品种数: {len(merged.columns)}')
    print(f'MERGE_MODE: {MERGE_MODE}')


if __name__ == '__main__':
    main()