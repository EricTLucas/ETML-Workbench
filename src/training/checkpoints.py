from dataclasses import replace
from pathlib import Path
import json
from data.manifest import write_json
from models import ModelConfig
from models.adapters.sklearn import dump_safe
from .artifacts import staged_directory, seal, verify_artifacts, require_files, environment


def load_checkpoint(path):
    root = Path(path).resolve()
    manifest = verify_artifacts(root,'training_checkpoint')
    require_files(manifest,['context.json','state.json','architecture.json','encoder.skops',
                            'preprocessing.json','environment.json'])
    context = json.loads((root/'context.json').read_text(encoding='utf-8'))
    backend = context['config']['backend']
    required = {'pytorch':['checkpoint.pt'],'tensorflow':['checkpoint.keras','best.npz']}
    if backend not in required:
        raise ValueError('Only PyTorch and TensorFlow checkpoints can be resumed')
    require_files(manifest,required[backend])
    saved = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    current = environment()
    for package in ('scikit-learn','torch' if backend=='pytorch' else 'tensorflow',
                    'numpy','keras' if backend=='tensorflow' else 'torch'):
        if saved.get(package) != current.get(package):
            raise ValueError(f'{package} differs from checkpoint; use its recorded environment')
    return context


def checkpoint_writer(directory,context,encoder,fitted):
    directory = Path(directory)
    def save(adapter,state):
        destination = directory/f"epoch-{state['epoch']:06d}"
        with staged_directory(destination) as staging:
            adapter._save_training(staging)
            dump_safe(encoder,staging/'encoder.skops')
            fitted.save(staging/'preprocessing.json')
            write_json(staging/'context.json',context)
            write_json(staging/'state.json',state)
            write_json(staging/'architecture.json',adapter.architecture)
            write_json(staging/'environment.json',environment())
            seal(staging,'training_checkpoint',{'epoch':state['epoch'],**context['origin']})
        return destination
    return save


def resume_training(workspace,checkpoint,*,epochs,reset_patience=False,**kwargs):
    from .runner import train_models
    context = load_checkpoint(checkpoint)
    config = ModelConfig(**context['config'])
    config = replace(config,params={**config.params,'epochs':epochs})
    origin = context['origin']
    return train_models(workspace,origin['dataset_id'],origin['task_id'],origin['preparation_run'],
                        configs=[config],metric=context['metric'],resume=checkpoint,
                        reset_patience=reset_patience,**kwargs)
