"""Run the tests with coverage in the parent and end-to-end subprocesses."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
config = root / '.coveragerc'
with tempfile.TemporaryDirectory(prefix='video2xy-coverage-') as folder:
    scratch = Path(folder)
    (scratch / 'sitecustomize.py').write_text(
        'import coverage\ncoverage.process_startup()\n', encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(scratch) + os.pathsep + env.get('PYTHONPATH', '')
    env['COVERAGE_PROCESS_START'] = str(config)
    env['COVERAGE_FILE'] = str(scratch / '.coverage')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'],
                           cwd=root, env=env)
    coverage_env = os.environ.copy()
    coverage_env['COVERAGE_FILE'] = str(scratch / '.coverage')
    for command in (['combine'], ['report'], ['json', '-o', str(root / 'coverage.json')]):
        subprocess.run([sys.executable, '-m', 'coverage', *command], cwd=root,
                       env=coverage_env, check=True)
    raise SystemExit(tests.returncode)
