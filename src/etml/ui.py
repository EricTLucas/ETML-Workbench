"""Launch the local project workbench without a frontend build step."""
import argparse
import threading
import webbrowser
import os


def launch(argv=None):
    parser = argparse.ArgumentParser(prog='workbench ui')
    parser.add_argument('--projects-dir', default=os.environ.get('ETML_PROJECTS_DIR', 'projects'), help='Same projects directory used by the CLI')
    parser.add_argument('--port', type=int, default=8501)
    parser.add_argument('--headless', action='store_true', help='Do not open a browser automatically')
    parser.add_argument('--max-upload-mb', type=int, default=512)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.max_upload_mb < 1:
        parser.error('Use a valid port and a positive upload limit')
    try:
        import uvicorn
        from .web.server import create_app
    except ImportError as exc:
        raise ImportError('Install the interface: pip install -e ".[ui]"') from exc
    app = create_app(args.projects_dir, max_bytes=args.max_upload_mb*1024**2)
    url = f'http://127.0.0.1:{args.port}'
    print(f'ETML Workbench: {url}\nProjects: {app.state.service.store.root}\nPress Ctrl+C to stop.')
    if not args.headless:
        timer = threading.Timer(1.2, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')
    return 0
