import re
from pathlib import Path

import pandas as pd


TOTAL_DIR = Path('./total')
INFO_PATH = Path('./Info.csv')
OUTPUT_PATH = Path('./Result/main_contract_history_summary.xlsx')
CONTRACT_PATTERN = re.compile(r'([A-Z]+)(\d{3,4})\.[A-Z]+$')


def extract_contract_code(mapping_ts_code: str) -> str | None:
    """从 mapping_ts_code 中提取合约月份编码，例如 CU2505.SHF -> 2505。"""
    if pd.isna(mapping_ts_code):
        return None

    match = CONTRACT_PATTERN.search(str(mapping_ts_code).strip().upper())
    if not match:
        return None
    return match.group(2)


def get_yearly_contract_sequence(df: pd.DataFrame) -> dict[str, list[str]]:
    """按年份提取主力合约切换序列，连续重复合约只保留一次。"""
    working = df[['trade_date', 'mapping_ts_code']].copy()
    working['trade_date'] = working['trade_date'].astype(str)
    working['year'] = working['trade_date'].str[:4]
    working['contract_code'] = working['mapping_ts_code'].map(extract_contract_code)
    working = working.dropna(subset=['contract_code'])
    working = working.sort_values('trade_date')

    yearly_sequence: dict[str, list[str]] = {}
    for year, group in working.groupby('year', sort=True):
        sequence: list[str] = []
        prev_contract = None
        for contract_code in group['contract_code']:
            if contract_code != prev_contract:
                sequence.append(contract_code)
                prev_contract = contract_code
        yearly_sequence[year] = sequence

    return yearly_sequence


def get_instrument_order() -> list[str]:
    """优先使用 Info.csv 的品种顺序；缺失时退化为 total 目录文件名顺序。"""
    if INFO_PATH.exists():
        info = pd.read_csv(INFO_PATH, encoding='utf-8-sig')
        return info['ts_code'].dropna().astype(str).tolist()

    return sorted(path.stem for path in TOTAL_DIR.glob('*.csv'))


def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    contract_sequence_map: dict[str, dict[str, str]] = {}
    contract_switch_count_map: dict[str, dict[str, int]] = {}
    all_years: set[str] = set()

    for csv_path in sorted(TOTAL_DIR.glob('*.csv')):
        ts_code = csv_path.stem
        df = pd.read_csv(csv_path, usecols=['trade_date', 'mapping_ts_code'])
        yearly_sequence = get_yearly_contract_sequence(df)

        contract_sequence_map[ts_code] = {}
        contract_switch_count_map[ts_code] = {}

        for year, sequence in yearly_sequence.items():
            all_years.add(year)
            contract_sequence_map[ts_code][year] = ' '.join(sequence)
            contract_switch_count_map[ts_code][year] = max(len(sequence) - 1, 0)

    year_columns = sorted(all_years)
    instrument_order = get_instrument_order()
    ordered_ts_codes = [ts_code for ts_code in instrument_order if ts_code in contract_sequence_map]
    remaining_ts_codes = sorted(set(contract_sequence_map) - set(ordered_ts_codes))
    ordered_ts_codes.extend(remaining_ts_codes)

    sequence_df = pd.DataFrame.from_dict(contract_sequence_map, orient='index')
    sequence_df = sequence_df.reindex(index=ordered_ts_codes, columns=year_columns)
    sequence_df.index.name = 'ts_code'

    switch_count_df = pd.DataFrame.from_dict(contract_switch_count_map, orient='index')
    switch_count_df = switch_count_df.reindex(index=ordered_ts_codes, columns=year_columns)
    switch_count_df = switch_count_df.astype('Int64')
    switch_count_df.index.name = 'ts_code'

    return sequence_df, switch_count_df


def main() -> None:
    if not TOTAL_DIR.exists():
        raise FileNotFoundError(f'total directory not found: {TOTAL_DIR}')

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sequence_df, switch_count_df = build_summary_tables()

    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        sequence_df.to_excel(writer, sheet_name='main_contracts')
        switch_count_df.to_excel(writer, sheet_name='switch_counts')

    print(f'输出完成: {OUTPUT_PATH}')


if __name__ == '__main__':
    main()