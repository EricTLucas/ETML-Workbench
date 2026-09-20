import argparse
from pathlib import Path
import subprocess
import sys


def launch(argv=None):
    parser=argparse.ArgumentParser(prog='workbench ui')
    parser.add_argument('--workspace',default='datasets')
    parser.add_argument('--port',type=int,default=8501)
    parser.add_argument('--headless',action='store_true')
    args=parser.parse_args(argv)
    try:
        import streamlit
    except ImportError as exc:
        raise ImportError('Install the interface: pip install -e ".[ui]"') from exc
    return subprocess.call([sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('app.py')),
        '--global.developmentMode','false',
        '--server.address','127.0.0.1','--server.port',str(args.port),
        '--server.headless',str(args.headless).lower(),'--browser.gatherUsageStats','false',
        '--theme.base','dark','--theme.primaryColor','#c084fc','--theme.backgroundColor','#100f17',
        '--theme.secondaryBackgroundColor','#211a2c','--','--workspace',str(Path(args.workspace).resolve())])
