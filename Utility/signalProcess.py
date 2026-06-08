"""Batch process RAW position files into transformed signals.

Input files are expected to follow this naming pattern:
    *_RAW_Position.csv

For each input file, the script can generate one or more outputs:
    *_STATE_MACHINE_Position.csv
    *_ZSCORE_Position.csv
    *_TANH_Position.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RAW_SUFFIX = "_RAW_Position.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAW position matrices to state machine, z-score, and tanh formats."
    )
    parser.add_argument("--input-dir", default="./Result", help="Directory containing RAW csv files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to input directory when omitted.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["state_machine", "zscore", "tanh"],
        default=["state_machine", "zscore", "tanh"],
        help="Methods to generate.",
    )
    parser.add_argument(
        "--zscore-window",
        type=int,
        default=60,
        help="Rolling window size for z-score.",
    )
    parser.add_argument(
        "--zscore-min-periods",
        type=int,
        default=20,
        help="Minimum periods for z-score rolling statistics.",
    )
    parser.add_argument(
        "--tanh-scale",
        type=float,
        default=1.0,
        help="Scale factor applied before tanh.",
    )
    parser.add_argument(
        "--sm-entry",
        type=float,
        default=0.0,
        help="State machine entry threshold.",
    )
    parser.add_argument(
        "--sm-exit",
        type=float,
        default=0.0,
        help="State machine exit threshold. Typically <= sm-entry.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV encoding for reading and writing.",
    )
    return parser.parse_args()


def list_raw_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob(f"*{RAW_SUFFIX}") if path.is_file())


def read_position_csv(path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, encoding=encoding)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    return df


def state_machine_transform(raw: pd.DataFrame, entry: float, exit_: float) -> pd.DataFrame:
    if exit_ > entry:
        raise ValueError("sm-exit should not be greater than sm-entry.")

    out = pd.DataFrame(index=raw.index, columns=raw.columns, dtype=float)
    for col in raw.columns:
        values = raw[col].to_numpy(dtype=float)
        states = np.zeros(values.shape[0], dtype=float)
        state = 0.0
        for i, x in enumerate(values):
            if np.isnan(x):
                states[i] = state
                continue

            if state == 0.0:
                if x > entry:
                    state = 1.0
                elif x < -entry:
                    state = -1.0
            elif state > 0.0:
                if x < -entry:
                    state = -1.0
                elif x < exit_:
                    state = 0.0
            else:
                if x > entry:
                    state = 1.0
                elif x > -exit_:
                    state = 0.0

            states[i] = state

        out[col] = states
    return out


def zscore_transform(raw: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    rolling_mean = raw.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = raw.rolling(window=window, min_periods=min_periods).std()
    z = (raw - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return z.fillna(0.0).clip(lower=-3.0, upper=3.0)


def tanh_transform(raw: pd.DataFrame, scale: float) -> pd.DataFrame:
    return np.tanh(raw * scale)


def output_paths(raw_file: Path, output_dir: Path) -> dict[str, Path]:
    base_name = raw_file.name[: -len(RAW_SUFFIX)]
    return {
        "state_machine": output_dir / f"{base_name}_STATE_MACHINE_Position.csv",
        "zscore": output_dir / f"{base_name}_ZSCORE_Position.csv",
        "tanh": output_dir / f"{base_name}_TANH_Position.csv",
    }


def process_one_file(
    raw_file: Path,
    output_dir: Path,
    methods: Iterable[str],
    encoding: str,
    zscore_window: int,
    zscore_min_periods: int,
    tanh_scale: float,
    sm_entry: float,
    sm_exit: float,
) -> None:
    raw = read_position_csv(raw_file, encoding=encoding)
    targets = output_paths(raw_file, output_dir)

    if "state_machine" in methods:
        sm = state_machine_transform(raw, entry=sm_entry, exit_=sm_exit)
        sm.to_csv(targets["state_machine"], encoding=encoding)

    if "zscore" in methods:
        z = zscore_transform(raw, window=zscore_window, min_periods=zscore_min_periods)
        z.to_csv(targets["zscore"], encoding=encoding)

    if "tanh" in methods:
        t = tanh_transform(raw, scale=tanh_scale)
        t.to_csv(targets["tanh"], encoding=encoding)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list_raw_files(input_dir)
    if not raw_files:
        print(f"No RAW files found in: {input_dir}")
        return

    print(f"Found {len(raw_files)} RAW files")
    for i, raw_file in enumerate(raw_files, start=1):
        process_one_file(
            raw_file=raw_file,
            output_dir=output_dir,
            methods=args.methods,
            encoding=args.encoding,
            zscore_window=args.zscore_window,
            zscore_min_periods=args.zscore_min_periods,
            tanh_scale=args.tanh_scale,
            sm_entry=args.sm_entry,
            sm_exit=args.sm_exit,
        )
        print(f"[{i}/{len(raw_files)}] processed: {raw_file.name}")

    print("Done")


if __name__ == "__main__":
    main()
