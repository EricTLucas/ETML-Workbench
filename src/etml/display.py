import json
import os
import re
import shlex


def command(*parts):
    """Copyable PowerShell commands on Windows, POSIX shell commands elsewhere."""
    def quote(value):
        text = str(value)
        if re.fullmatch(r'[A-Za-z0-9_./:=\\-]+',text):
            return text
        return "'"+text.replace("'","''")+"'" if os.name=='nt' else shlex.quote(text)
    return ' '.join(quote(p) for p in parts)


def emit(payload,args,*,steps=(),table=None):
    payload = dict(payload)
    if steps:
        payload['next_steps'] = [{'description':description,'command':text} for description,text in steps]
    if args.json:
        print(json.dumps(payload,ensure_ascii=False,allow_nan=False))
        return
    if table is not None:
        headers,rows = table
        rows = [[str(value) for value in row] for row in rows]
        widths = [max(len(str(h)),*(len(row[i]) for row in rows)) if rows else len(str(h)) for i,h in enumerate(headers)]
        print('  '.join(str(h).ljust(widths[i]) for i,h in enumerate(headers)))
        for row in rows:
            print('  '.join(value.ljust(widths[i]) for i,value in enumerate(row)))
        for key in ('preparation_run','training_run','metric','winner','bundle','directory'):
            if payload.get(key) is not None:
                print(f'{key.replace("_"," ").capitalize()}: {payload[key]}')
    else:
        print(json.dumps({k:v for k,v in payload.items() if k!='next_steps'},indent=2,ensure_ascii=False,allow_nan=False))
    if steps and not args.quiet:
        print('\nSuggested next steps:')
        for description,text in steps:
            print(f'  {description}\n    {text}')
