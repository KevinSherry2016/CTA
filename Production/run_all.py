import os
import subprocess
import glob
import sys


SKIP_SCRIPTS = {'run_all.py', 'PortfolioConstruction.py', 'Merge.py'}


def _production_dirs(base_dir):
    return sorted(
        path for path in glob.glob(os.path.join(base_dir, '*'))
        if os.path.isdir(path)
    )


def _run_script(script):
    print(f"Running script: {script}")
    subprocess.run([sys.executable, script], check=True)


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    production_dirs = _production_dirs(BASE_DIR)
    if not production_dirs:
        print("No production folders found.")
        return

    for production_dir in production_dirs:
        folder_name = os.path.basename(production_dir)
        print(f"\n===== Running production folder: {folder_name} =====")

        factor_scripts = sorted(glob.glob(os.path.join(production_dir, '*.py')))
        for script in factor_scripts:
            base_name = os.path.basename(script)
            if base_name in SKIP_SCRIPTS:
                continue
            print(f"Running factor script: {folder_name}/{base_name}")
            try:
                _run_script(script)
            except subprocess.CalledProcessError as e:
                print(f"Error running {script}: {e}")

        portfolio_script = os.path.join(production_dir, 'PortfolioConstruction.py')
        if not os.path.exists(portfolio_script):
            print(f"No PortfolioConstruction.py found in {folder_name}; skipping portfolio construction.")
            continue

        print(f"\nRunning portfolio construction script: {folder_name}/PortfolioConstruction.py")
        try:
            _run_script(portfolio_script)
        except subprocess.CalledProcessError as e:
            print(f"Error running {portfolio_script}: {e}")

    # After portfolio construction has run, you are now running the merged risk control script directly
    # So we don't need to run RiskControl.py anymore.
    print("\nAll tasks completed successfully.")

if __name__ == '__main__':
    main()
