from pathlib import Path
import py_compile
import traceback

p = Path('src') / 'trustworthy_resume' / 'experiment.py'
print('Checking', p)
try:
    py_compile.compile(str(p), doraise=True)
    print('COMPILE_OK')
except Exception as e:
    print(type(e).__name__, e)
    traceback.print_exc()