"""Batch runner for factor generation -> signal processing -> evaluation.

Config-only usage:
- Edit PIPELINE_CONFIG in this file.
- Run: python Utility/run_factor_pipeline.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
import re

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
FACTOR_DIR = ROOT_DIR / "Factor"
UTILITY_DIR = ROOT_DIR / "Utility"
SIGNAL_PROCESS_SCRIPT = UTILITY_DIR / "signalProcess.py"
FACTOR_EVAL_SCRIPT = UTILITY_DIR / "factorEvaluation.py"
SUMMARY_METRICS_PATH = ROOT_DIR / "Evaluate" / "all_metrics_summary.csv"

PIPELINE_CONFIG = {
    "factors": [],
    "n_list": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
    # "n_list": [10,20,30],
    "backtest_start_date": 20180101,
    "custom_symbols": [],
    "sector_list": ['Energy','Agriculture','Ferrous','Precious','NonFerrous'],
    "run_all": True,
    "run_custom_symbols": False,
    "excluded_symbols": [],
    "stop_on_error": False,
}


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_factor_overrides(
    module,
    n_list: list[int],
    backtest_start_date: str,
    sector_list: list[str],
    custom_symbols: list[str],
    excluded_symbols: list[str],
) -> None:
    if n_list and hasattr(module, "N_LIST") and not hasattr(module, "PARAM_LIST"):
        module.N_LIST = list(dict.fromkeys(n_list))

    module.BACKTEST_START_DATE = str(backtest_start_date)

    if hasattr(module, "SECTOR_LIST"):
        module.SECTOR_LIST = list(dict.fromkeys(sector_list))

    if not custom_symbols:
        pass
    else:
        module.CUSTOM_GROUPS = [
            {
                "name": "CustomSymbols",
                "symbols": list(dict.fromkeys(custom_symbols)),
            }
        ]

    if excluded_symbols:
        module.EXCLUDED_SYMBOLS = list(dict.fromkeys(excluded_symbols))


def _run_one_factor(
    script_path: Path,
    n_list: list[int],
    backtest_start_date: str,
    sector_list: list[str],
    custom_symbols: list[str],
    run_all: bool,
    run_custom_symbols: bool,
    excluded_symbols: list[str],
) -> None:
    module = _load_module(script_path)
    _apply_factor_overrides(
        module,
        n_list,
        backtest_start_date,
        sector_list,
        custom_symbols,
        excluded_symbols,
    )

    # Prefer config-driven execution path for template-based factor scripts.
    if all(hasattr(module, name) for name in ["_run_and_save", "FACTOR_NAME", "_safe_name"]):
        info_path = ROOT_DIR / "Info.csv"
        info = pd.read_csv(info_path, encoding="utf-8-sig")

        excluded_sectors = [s.lower() for s in getattr(module, "EXCLUDED_SECTORS", [])]
        valid_info = info[~info["sector"].str.lower().isin(excluded_sectors)]
        excluded_symbol_list = getattr(module, "EXCLUDED_SYMBOLS", [])
        if excluded_symbol_list:
            valid_info = valid_info[~valid_info["ts_code"].isin(excluded_symbol_list)]

        valid_symbol_set = set(valid_info["ts_code"].tolist())
        market_data_path = "./main_contract/"
        print(f"Loading {module.FACTOR_NAME} data...")

        if hasattr(module, "PARAM_LIST"):
            params = list(module.PARAM_LIST)
        elif hasattr(module, "N_LIST"):
            params = list(module.N_LIST)
        else:
            raise AttributeError(f"{script_path.name} has neither N_LIST nor PARAM_LIST")

        for param in params:
            if run_all:
                module._run_and_save("ALL", sorted(valid_symbol_set), market_data_path, param)

            if sector_list:
                for sector in sector_list:
                    sector_symbols = valid_info[valid_info["sector"] == sector]["ts_code"].tolist()
                    if sector_symbols:
                        module._run_and_save(sector, sector_symbols, market_data_path, param)

            if run_custom_symbols:
                for group in getattr(module, "CUSTOM_GROUPS", []):
                    raw_symbols = group.get("symbols", [])
                    if isinstance(raw_symbols, str):
                        raw_symbols = [raw_symbols]
                    group_symbols = sorted(set(s for s in raw_symbols if s in valid_symbol_set))
                    merged_name = "_".join(group_symbols) if group_symbols else group.get("name", "CustomGroup")
                    module._run_and_save(merged_name, group_symbols, market_data_path, param)
        return

    if not hasattr(module, "main"):
        raise AttributeError(f"{script_path.name} has no main()")
    module.main()


def _run_python_script(
    script_path: Path,
    script_args: list[str] | None = None,
) -> None:
    cmd = [sys.executable, str(script_path)]
    if script_args:
        cmd.extend(script_args)
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)


def _collect_factor_names(factor_scripts: list[Path]) -> list[str]:
    names: set[str] = set()
    for script_path in factor_scripts:
        names.add(script_path.stem)
        try:
            module = _load_module(script_path)
            factor_name = getattr(module, "FACTOR_NAME", "")
            if isinstance(factor_name, str) and factor_name:
                names.add(factor_name)
        except Exception:
            continue
    return sorted(names, key=len, reverse=True)


def _parse_strategy_file(
    strategy_file: str,
    factor_names: list[str],
    sector_names_lower: set[str],
) -> dict[str, str]:
    stem = Path(strategy_file).stem
    if stem.endswith("_Position"):
        stem = stem[: -len("_Position")]

    parts = stem.split("_")
    if len(parts) < 3:
        return {
            "因子名称": stem,
            "回测品种": "自定义品种",
            "参数": "",
            "信号方式": "",
        }

    signal_token = parts[-1]
    param_token = parts[-2]
    prefix = "_".join(parts[:-2])

    factor_name = ""
    target_name = ""
    for candidate in factor_names:
        if prefix == candidate:
            factor_name = candidate
            target_name = ""
            break
        if prefix.startswith(candidate + "_"):
            factor_name = candidate
            target_name = prefix[len(candidate) + 1 :]
            break

    if not factor_name:
        factor_name = prefix

    target_lower = target_name.lower()
    if target_lower == "all":
        instrument_group = "all"
    elif target_lower in sector_names_lower:
        instrument_group = "sector"
    else:
        instrument_group = "自定义品种"

    signal_map = {
        "RAW": "raw",
        "ZSCORE": "zscore",
        "STATE_MACHINE": "state_machine",
        "TANH": "tanh",
    }
    signal_method = signal_map.get(signal_token.upper(), signal_token.lower())

    if not re.match(r"^N\d+$", param_token.upper()):
        param_token = param_token

    return {
        "因子名称": factor_name,
        "回测品种": instrument_group,
        "参数": param_token,
        "信号方式": signal_method,
    }


def _split_summary_first_column(summary_path: Path, factor_scripts: list[Path]) -> None:
    if not summary_path.exists():
        print(f"Summary file not found, skip split: {summary_path}")
        return

    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if "strategyFile" not in df.columns:
        print("Column strategyFile not found, skip split.")
        return

    info = pd.read_csv(ROOT_DIR / "Info.csv", encoding="utf-8-sig")
    sector_names_lower = set(info["sector"].dropna().astype(str).str.lower().tolist())
    factor_names = _collect_factor_names(factor_scripts)

    expected_columns = ["因子名称", "回测品种", "参数", "信号方式"]
    if set(expected_columns).issubset(df.columns):
        ordered = expected_columns + [c for c in df.columns if c not in set(expected_columns)]
        df = df[ordered]
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Summary already split, columns reordered: {summary_path}")
        return

    parsed_rows = [
        _parse_strategy_file(str(name), factor_names, sector_names_lower)
        for name in df["strategyFile"].astype(str).tolist()
    ]
    parsed_df = pd.DataFrame(parsed_rows)

    merged = pd.concat([parsed_df, df], axis=1)
    ordered = ["因子名称", "回测品种", "参数", "信号方式"] + [
        c for c in merged.columns if c not in {"因子名称", "回测品种", "参数", "信号方式"}
    ]
    merged = merged[ordered]
    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Split summary columns written: {summary_path}")


def main() -> None:
    factors = PIPELINE_CONFIG["factors"]
    n_list = PIPELINE_CONFIG["n_list"]
    sector_list = PIPELINE_CONFIG["sector_list"]
    custom_symbols = PIPELINE_CONFIG["custom_symbols"]
    run_all = PIPELINE_CONFIG["run_all"]
    run_custom_symbols = PIPELINE_CONFIG["run_custom_symbols"]
    excluded_symbols = PIPELINE_CONFIG["excluded_symbols"]
    stop_on_error = PIPELINE_CONFIG["stop_on_error"]
    backtest_start_date = str(PIPELINE_CONFIG["backtest_start_date"])

    if not FACTOR_DIR.exists():
        raise FileNotFoundError(f"Factor directory not found: {FACTOR_DIR}")
    if not SIGNAL_PROCESS_SCRIPT.exists():
        raise FileNotFoundError(f"signalProcess script not found: {SIGNAL_PROCESS_SCRIPT}")
    if not FACTOR_EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"factorEvaluation script not found: {FACTOR_EVAL_SCRIPT}")

    factor_scripts = sorted(path for path in FACTOR_DIR.glob("*.py") if path.is_file())
    requested = {name.strip() for name in factors if name.strip()}

    selected_scripts: list[Path] = []
    if not requested:
        selected_scripts = factor_scripts
    else:
        for script_path in factor_scripts:
            module = _load_module(script_path)
            if script_path.stem in requested or getattr(module, "FACTOR_NAME", "") in requested:
                selected_scripts.append(script_path)

    if not selected_scripts:
        requested_text = ", ".join(factors) if factors else "<none>"
        raise ValueError(f"No factor scripts matched requested names: {requested_text}")

    print(f"Selected factors: {len(selected_scripts)}")
    failed: list[str] = []

    for idx, script_path in enumerate(selected_scripts, start=1):
        print(f"[Factor {idx}/{len(selected_scripts)}] Running {script_path.stem} ...")
        try:
            _run_one_factor(
                script_path,
                n_list,
                backtest_start_date,
                sector_list,
                custom_symbols,
                run_all,
                run_custom_symbols,
                excluded_symbols,
            )
        except Exception as exc:
            msg = f"{script_path.name}: {exc}"
            failed.append(msg)
            print(f"Failed: {msg}")
            if stop_on_error:
                raise

    if failed:
        print("Some factor scripts failed:")
        for item in failed:
            print(f"  - {item}")

    print("Running signalProcess ...")
    _run_python_script(SIGNAL_PROCESS_SCRIPT)

    print("Running factorEvaluation ...")
    _run_python_script(FACTOR_EVAL_SCRIPT)

    print("Formatting all_metrics_summary ...")
    _split_summary_first_column(SUMMARY_METRICS_PATH, factor_scripts)

    print("Pipeline finished")


if __name__ == "__main__":
    main()
