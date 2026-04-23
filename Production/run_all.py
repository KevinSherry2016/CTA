import os
import subprocess
import glob

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PRODUCTION_DIR = os.path.join(BASE_DIR, '20260422')
    MERGE_SCRIPT = os.path.join(PRODUCTION_DIR, 'Merge.py')

    # Get all python scripts in the production directory
    factor_scripts = glob.glob(os.path.join(PRODUCTION_DIR, '*.py'))
    
    # Run each factor script
    for script in factor_scripts:
        base_name = os.path.basename(script)
        if base_name in ['run_all.py', 'Merge.py']:
            continue
        print(f"Running factor script: {base_name}")
        try:
            subprocess.run(['python', script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    # After all factors have run, run the merge script
    print(f"\nRunning merge script: {os.path.basename(MERGE_SCRIPT)}")
    try:
        subprocess.run(['python', MERGE_SCRIPT], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {MERGE_SCRIPT}: {e}")

    print("\nAll tasks completed successfully.")

if __name__ == '__main__':
    main()
