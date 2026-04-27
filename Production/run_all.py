import os
import subprocess
import glob

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PRODUCTION_DIR = os.path.join(BASE_DIR, '20260422')
    PORTFOLIO_SCRIPT = os.path.join(PRODUCTION_DIR, 'PortfolioConstruction.py')
    RISK_SCRIPT = os.path.join(PRODUCTION_DIR, 'RiskControl.py')

    # Get all python scripts in the production directory
    factor_scripts = glob.glob(os.path.join(PRODUCTION_DIR, '*.py'))
    
    # Run each factor script
    for script in factor_scripts:
        base_name = os.path.basename(script)
        if base_name in ['run_all.py', 'PortfolioConstruction.py', 'RiskControl.py', 'Merge.py']:
            continue
        print(f"Running factor script: {base_name}")
        try:
            subprocess.run(['python', script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    # After all factors have run, run the portfolio construction script
    print(f"\nRunning portfolio construction script: {os.path.basename(PORTFOLIO_SCRIPT)}")
    try:
        subprocess.run(['python', PORTFOLIO_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {PORTFOLIO_SCRIPT}: {e}")

    # After portfolio construction has run, run the risk control script
    print(f"\nRunning risk control script: {os.path.basename(RISK_SCRIPT)}")
    try:
        subprocess.run(['python', RISK_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {RISK_SCRIPT}: {e}")

    print("\nAll tasks completed successfully.")

if __name__ == '__main__':
    main()
