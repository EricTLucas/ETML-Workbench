"""Install catalog-owned optional model requirements in the running interpreter."""
import importlib
from importlib.metadata import version, PackageNotFoundError
import subprocess
import sys
import threading
from packaging.requirements import Requirement

EXTRAS = {
    'xgboost': ['xgboost>=2.0'], 'lightgbm': ['lightgbm>=4.0'], 'catboost': ['catboost>=1.2'],
    'pytorch': ['torch>=2.6'], 'tensorflow': ['tensorflow>=2.18'],
    'vision': ['torch>=2.6','torchvision>=0.21','Pillow>=10'],
    'text': ['torch>=2.6','transformers>=4.45,<6','safetensors>=0.4','sentencepiece>=0.2'],
    'timeseries': ['statsmodels>=0.14'], 'recommendation': ['implicit>=0.7'],
}
_LOCK = threading.Lock()

def missing_requirements(extra):
    if not extra: return []
    if extra not in EXTRAS: raise ValueError('Unknown catalog dependency group')
    missing=[]
    for text in EXTRAS[extra]:
        requirement=Requirement(text)
        try: installed=version(requirement.name)
        except PackageNotFoundError: missing.append(text); continue
        if installed not in requirement.specifier:
            raise ValueError(f'{requirement.name} {installed} is incompatible with {text}. Update the environment and restart the workbench.')
    return missing

def ensure_dependencies(key, progress=lambda event: None):
    from .catalog import CATALOG
    entry=CATALOG[key]
    with _LOCK:
        missing=missing_requirements(entry.extra)
        if not missing: return
        progress({'stage':'dependencies','message':'Installing model packages: '+', '.join(missing)})
        try:
            result=subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',*missing],
                capture_output=True,text=True,timeout=1800)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise ValueError('Package installation could not finish. Check internet access and pip, then retry training.') from exc
        if result.returncode:
            raise ValueError('Package installation failed. Check platform support, disk space and pip permissions, then retry. '+result.stderr[-2000:])
        importlib.invalidate_caches()
        if missing_requirements(entry.extra): raise ValueError('Installation finished but packages are unavailable. Restart the workbench and retry.')
        progress({'stage':'dependencies','message':'Model packages installed; continuing training'})
