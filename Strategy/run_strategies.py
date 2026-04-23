import glob, os, subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scripts = [f for f in glob.glob(os.path.join(BASE_DIR, '*.py')) if not f.endswith('Template.py') and not os.path.basename(f).startswith('generator') and not f.endswith('run_strategies.py')]

for s in scripts:
    print(f"Running {os.path.basename(s)}...")
    subprocess.run(['C:/Users/liukaicheng/anaconda3/python.exe', s])
print("All strategies executed.")
