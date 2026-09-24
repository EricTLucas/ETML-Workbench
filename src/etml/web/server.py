"""Loopback-only HTTP interface. Mutations use background jobs."""
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
from tempfile import TemporaryDirectory
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from data.manifest import resolve_inside
from data.sources import SKLEARN_DATASETS
from preprocessing.transforms import available_transforms
from .service import Service


def create_app(projects_dir='projects', *, max_bytes=512*1024**2):
    service = Service(projects_dir, max_bytes=max_bytes)
    token = secrets.token_urlsafe(32)
    static = Path(__file__).with_name('static')

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title='ETML Workbench', docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.service = service
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])

    @app.middleware('http')
    async def local_session(request, call_next):
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return JSONResponse({'detail': 'Cross-origin requests are not permitted.'}, status_code=403)
        path = request.url.path
        if path.startswith('/api/'):
            if not secrets.compare_digest(request.headers.get('x-etml-token', ''), token):
                return JSONResponse({'detail': 'Reload the workbench to renew your session.'}, status_code=403)
        if (path.startswith('/reports/') or path.startswith('/downloads/')) and not secrets.compare_digest(request.cookies.get('etml-session', ''), token):
            return JSONResponse({'detail': 'Open the workbench first.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'" + (" 'unsafe-inline'" if path.startswith('/reports/') else '') +
            "; frame-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'")
        return response

    @app.exception_handler(ValueError)
    @app.exception_handler(FileNotFoundError)
    @app.exception_handler(FileExistsError)
    @app.exception_handler(KeyError)
    def invalid(request, exc):
        return JSONResponse({'detail': str(exc)}, status_code=400)

    @app.get('/', response_class=HTMLResponse)
    def index():
        response = HTMLResponse((static/'index.html').read_text(encoding='utf-8').replace('__TOKEN__', token))
        response.set_cookie('etml-session', token, httponly=True, samesite='strict')
        return response

    app.mount('/static', StaticFiles(directory=static), name='static')

    @app.get('/api/bootstrap')
    def bootstrap():
        return {'projects': [p.name for p in service.store.list()], 'library': SKLEARN_DATASETS,
                'transforms': available_transforms(), 'max_upload_bytes': max_bytes}

    @app.post('/api/projects')
    def create(payload: dict):
        project = service.store.create(payload['name'])
        return {'name': project.name}

    @app.get('/api/projects/{name}')
    def overview(name: str, dataset_id: str | None = None):
        return service.overview(name, dataset_id)

    @app.get('/api/jobs/{job_id}')
    def job(job_id: str):
        return service.job(job_id)

    @app.post('/api/projects/{name}/import')
    def import_data(name: str, payload: dict):
        return service.submit(name, 'Importing data and building your overview', lambda: service.import_data(name, payload))

    @app.post('/api/projects/{name}/upload')
    async def upload(name: str, file: UploadFile = File(...), analyze: bool = Form(True)):
        service.store.get(name)
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in {'.csv', '.tsv', '.parquet', '.json', '.jsonl', '.ndjson'}:
            raise HTTPException(400, 'Choose a CSV, TSV, Parquet, JSON or JSONL dataset.')
        temp = TemporaryDirectory(prefix='etml-upload-')
        filename = Path((file.filename or 'upload'+suffix).replace('\\', '/')).name
        path = Path(temp.name)/filename
        try:
            size = 0
            with path.open('wb') as output:
                while block := await file.read(1024*1024):
                    size += len(block)
                    if size > max_bytes:
                        raise HTTPException(413, 'Upload exceeds the configured size limit.')
                    output.write(block)
            def run():
                try:
                    return service.import_data(name, {'source': 'path', 'path': str(path), 'analyze': analyze})
                finally:
                    temp.cleanup()
            return service.submit(name, 'Importing data and building your overview', run)
        except Exception:
            temp.cleanup()
            raise
        finally:
            await file.close()

    @app.post('/api/projects/{name}/analyze')
    def analyze(name: str, payload: dict):
        return service.submit(name, 'Generating the data overview', lambda: service.analyze(name, payload['dataset_id']))

    @app.post('/api/projects/{name}/suggest')
    def suggest(name: str, payload: dict):
        return service.submit(name, 'Finding preprocessing recommendations', lambda: service.suggest(name, payload))

    @app.post('/api/projects/{name}/preview')
    def preview(name: str, payload: dict):
        return service.submit(name, 'Fitting the recipe and previewing changes', lambda: service.preview(name, payload))

    @app.post('/api/projects/{name}/save')
    def save(name: str, payload: dict):
        return service.submit(name, 'Saving a processed version', lambda: service.save(name, payload))

    @app.post('/api/projects/{name}/split')
    def split(name: str, payload: dict):
        return service.submit(name, 'Creating training, validation and test splits', lambda: service.split(name, payload))

    from .models import ModelService
    models = ModelService(service)

    @app.get('/api/projects/{name}/models')
    def model_list(name: str):
        return models.overview(name)

    @app.get('/api/projects/{name}/models/compare')
    def compare_models(name: str):
        return models.compare(name)

    @app.get('/api/projects/{name}/models/{model}')
    def model_details(name: str, model: str):
        return models.details(name, model)

    @app.post('/api/projects/{name}/models/train')
    def train_model(name: str, payload: dict):
        return service.submit(name, 'Training model', lambda progress: models.train(name, payload, progress), with_progress=True)

    @app.post('/api/projects/{name}/models/{model}/predict')
    def predict_model(name: str, model: str, payload: dict):
        return service.submit(name, 'Running prediction', lambda: models.predict(name, model, payload))

    @app.post('/api/projects/{name}/models/{model}/predict-csv')
    async def predict_csv(name: str, model: str, file: UploadFile = File(...)):
        service.store.get(name)
        if Path(file.filename or '').suffix.lower() != '.csv':
            await file.close()
            raise HTTPException(400, 'Choose a CSV file.')
        temp=TemporaryDirectory(prefix='etml-predict-')
        source=Path(temp.name)/'input.csv'
        try:
            size=0
            with source.open('wb') as output:
                while block := await file.read(1024*1024):
                    size+=len(block)
                    if size>max_bytes: raise HTTPException(413,'Upload exceeds the configured size limit.')
                    output.write(block)
            def run():
                try: return models.predict_csv(name,model,source)
                finally: temp.cleanup()
            return service.submit(name,'Predicting CSV rows',run)
        except Exception:
            temp.cleanup(); raise
        finally:
            await file.close()

    @app.post('/api/projects/{name}/prediction-exports/{export_id}/filter')
    def filter_predictions(name: str, export_id: str, payload: dict):
        from .csv_exports import filter_csv
        project=service.store.get(name)
        return service.submit(name,'Filtering prediction export',lambda:filter_csv(project,export_id,payload))

    @app.post('/api/projects/{name}/analysis-summary')
    def analysis_summary(name: str, payload: dict):
        from .analysis_report import export_analysis
        return service.submit(name,'Building analysis summary',lambda:export_analysis(models,name,payload))

    @app.get('/downloads/{name}/{export_id}/analysis.html')
    def analysis_download(name: str, export_id: str):
        from data.manifest import validate_name
        project=service.store.get(name)
        path=resolve_inside(project.directory/'exports',validate_name(export_id))/'analysis.html'
        if not path.is_file():raise HTTPException(404,'Analysis report not found')
        return FileResponse(path,filename=name+'-analysis.html',media_type='text/html')

    @app.get('/downloads/{name}/{export_id}/predictions.csv')
    def prediction_download(name: str, export_id: str):
        from data.manifest import validate_name
        project=service.store.get(name)
        path=resolve_inside(project.directory/'exports',validate_name(export_id))/'predictions.csv'
        if not path.is_file(): raise HTTPException(404,'Predictions not found')
        return FileResponse(path,filename='predictions.csv',media_type='text/csv')

    @app.post('/api/projects/{name}/models/{model}/evaluate')
    def evaluate_model(name: str, model: str):
        return service.submit(name, 'Evaluating held-out test data', lambda: models.evaluate(name, model))

    @app.post('/api/projects/{name}/models/{model}/export')
    def export_model(name: str, model: str, payload: dict):
        return service.submit(name, 'Exporting model', lambda: models.export(name, model, payload))

    @app.get('/downloads/{name}/{export_id}')
    def download(name: str, export_id: str):
        from data.manifest import validate_name
        project = service.store.get(name)
        root = resolve_inside(project.directory/'exports', validate_name(export_id))
        archive = root/'model.zip'
        if not archive.is_file():
            raise HTTPException(404, 'Export not found')
        return FileResponse(archive, filename=export_id+'.zip', media_type='application/zip')

    @app.get('/reports/{name}/{dataset_id}/{asset:path}')
    def report(name: str, dataset_id: str, asset: str):
        dataset = service.store.get(name).workspace.get(dataset_id)
        path = resolve_inside(dataset.profiles_dir, asset)
        if path.suffix.lower() not in {'.html', '.png', '.jpg', '.jpeg', '.svg', '.css', '.js'} or not path.is_file():
            raise HTTPException(404, 'Report asset not found')
        return FileResponse(path)

    return app
