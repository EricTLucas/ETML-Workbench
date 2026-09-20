import argparse
import sys
from .display import emit,command


COMMANDS={'import-split','from-sklearn','from-huggingface','from-url','from-openml'}


def main(argv=None):
    parser=argparse.ArgumentParser(prog='workbench datasets')
    subs=parser.add_subparsers(dest='command',required=True)
    split=subs.add_parser('import-split',help='Preserve existing training/test/validation files')
    split.add_argument('--train',required=True); split.add_argument('--test'); split.add_argument('--validation')
    sklearn=subs.add_parser('from-sklearn'); sklearn.add_argument('name')
    hf=subs.add_parser('from-huggingface'); hf.add_argument('repository')
    hf.add_argument('--config'); hf.add_argument('--split',default='train'); hf.add_argument('--revision')
    hf.add_argument('--train-split'); hf.add_argument('--test-split'); hf.add_argument('--validation-split')
    hf.add_argument('--columns',nargs='+')
    url=subs.add_parser('from-url'); url.add_argument('url'); url.add_argument('--filename')
    openml=subs.add_parser('from-openml'); openml.add_argument('data_id',type=int)
    for p in (split,sklearn,hf,url,openml):
        p.add_argument('--id',dest='dataset_id'); p.add_argument('--workspace',default='datasets')
        p.add_argument('--max-memory-mb',type=int,default=512); p.add_argument('--max-rows',type=int,default=200_000)
        p.add_argument('--json',action='store_true'); p.add_argument('--quiet',action='store_true')
    args=parser.parse_args(argv)
    try:
        from data import DatasetWorkspace
        from data.sources import import_sklearn,import_openml,import_url,import_huggingface
        ws=DatasetWorkspace(args.workspace)
        options={'dataset_id':args.dataset_id,'max_bytes':args.max_memory_mb*1024**2}
        if args.command=='import-split':
            dataset=ws.import_presplit(args.train,test=args.test,validation=args.validation,**options)
        elif args.command=='from-sklearn': dataset=import_sklearn(ws,args.name,max_rows=args.max_rows,**options)
        elif args.command=='from-openml': dataset=import_openml(ws,args.data_id,max_rows=args.max_rows,**options)
        elif args.command=='from-url': dataset=import_url(ws,args.url,filename=args.filename,**options)
        else:
            splits={key:value for key,value in {'train':args.train_split,'test':args.test_split,'validation':args.validation_split}.items() if value}
            if splits: splits.setdefault('train','train')
            dataset=import_huggingface(ws,args.repository,config=args.config,split=args.split,splits=splits or None,
                revision=args.revision,columns=args.columns,max_rows=args.max_rows,**options)
        payload={'status':'ok','dataset_id':dataset.dataset_id,'directory':str(dataset.directory),
                 'split_files':dataset.manifest.split_files,'provenance':dataset.manifest.provenance}
        steps=[('Inspect training data and propose preprocessing',command('workbench','preprocess','prepare',dataset.dataset_id,'--workspace',args.workspace))]
        if dataset.manifest.provenance.get('target'):
            provenance=dataset.manifest.provenance
            steps.append(('Create a prediction task',command('workbench','tasks','create',dataset.dataset_id,'prediction',
                '--target',provenance['target'],'--type',provenance['task_type'],'--workspace',args.workspace)))
        emit(payload,args,steps=steps)
        return 0
    except (ValueError,TypeError,OSError,ImportError,RuntimeError,KeyError) as exc:
        print(f'error: {exc}',file=sys.stderr); return 2
