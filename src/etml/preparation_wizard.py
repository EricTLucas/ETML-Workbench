"""Human-facing recipe review and split selection for a project's current dataset."""
from dataclasses import replace
import json
from pathlib import Path
from preprocessing import Recipe, Step
from preprocessing.transforms import available_transforms
from preprocessing.validation import validate_recipe
from splitting import SplitConfig
from workflows.project_preparation import ProjectPreparation
from .display import command


def _columns(prompt, columns, ask):
    for i, column in enumerate(columns,1):
        print(f'  {i}. {column}')
    while True:
        raw = ask(prompt)
        if not raw:
            return []
        try:
            indexes = [int(x.strip())-1 for x in raw.split(',')]
            if len(set(indexes)) != len(indexes) or any(i<0 or i>=len(columns) for i in indexes):
                raise ValueError
            return [columns[i] for i in indexes]
        except ValueError:
            print('Use column numbers separated by commas, or Enter for none.')


def _params(ask):
    while True:
        try:
            value = json.loads(ask('Parameters as a JSON object (Enter for {}): ') or '{}')
            if not isinstance(value,dict):
                raise ValueError('Parameters must be a JSON object')
            return value
        except ValueError as exc:
            print(f'Invalid parameters: {exc}')


def _review(recipe, columns, choose, ask):
    reviewed = []
    for step in recipe.steps:
        print('\n'+step.operation.replace('_',' ').capitalize()+' — '+', '.join(step.columns))
        if step.reason:
            print(step.reason)
        if step.params:
            print('Settings: '+', '.join(f'{k}={v}' for k,v in step.params.items()))
        if not step.enabled or step.status=='rejected':
            print('Currently disabled in this recipe.')
        choice = choose('Use this change?', ['Keep', 'Change settings', 'Skip'])
        if choice == 2:
            reviewed.append(step.reject())
            continue
        if choice == 1:
            if step.operation == 'fill_missing':
                strategy = ['mean','median','mode','constant'][choose('Missing-value strategy',
                    ['Mean','Median','Most common value (mode)','Constant value'])]
                params = {'strategy':strategy}
                if strategy=='constant':
                    while True:
                        try:
                            value = json.loads(ask('Fill value as JSON (e.g. 0 or "Unknown"): '))
                            if value is None or not isinstance(value,(str,int,float,bool)):
                                raise ValueError('Use a non-null scalar')
                            params['value'] = value
                            break
                        except ValueError as exc:
                            print(str(exc))
                step = step.edit(params=params)
            else:
                step = step.edit(params=_params(ask))
        reviewed.append(replace(step,enabled=True).approve())
    while choose('\nAdditional changes', ['Continue to preview','Add a custom transformation']) == 1:
        operations = list(available_transforms())
        operation = operations[choose('Transformation',operations)]
        current_columns = list(validate_recipe(Recipe(tuple(reviewed)),columns).output_columns)
        names = _columns('Column numbers (comma-separated): ', current_columns, ask)
        if not names:
            print('No columns selected; nothing added.')
            continue
        print('Parameter names: '+', '.join(available_transforms()[operation]))
        candidate = Step(operation,tuple(names),_params(ask)).approve()
        try:
            validate_recipe(Recipe(tuple(reviewed+[candidate])),columns)
        except (ValueError,TypeError) as exc:
            print(f'Cannot add this step: {exc}')
            continue
        reviewed.append(candidate)
    result = Recipe(tuple(reviewed),name='Reviewed project preprocessing')
    validate_recipe(result,columns)
    return result


def _configure(service, choose, ask):
    names = list(service.schema())
    suggested = service.dataset.manifest.provenance.get('target')
    print('\nChoose the column you will predict so preprocessing leaves its values untouched.')
    target = names[choose('Target column',[n+(' (dataset target)' if n==suggested else '') for n in names])]
    task_type = ['classification','regression'][choose('Prediction type',
        ['Classification — predict a category','Regression — predict a number'])]
    remaining = [n for n in names if n!=target]
    excluded = _columns('Exclude ID/unwanted column numbers (Enter for none): ',remaining,ask)
    policy = ['error','drop'][choose('If target values are missing',
        ['Stop and let me fix them','Exclude those rows when splitting'])]
    service.configure(target,task_type,excluded,policy)


def _recipe(service, choose, ask):
    task = service.task()
    columns = [n for n in service.schema() if n not in {task.target,*task.excluded_columns}]
    choice = choose('\nPreprocessing recipe', ['Review suggested changes','Load custom recipe JSON',
                                               'Keep features unchanged','Save for later'])
    if choice == 3:
        return False
    if choice == 0:
        print('Preparing suggestions...')
        recipe = service.suggest()
        if not recipe.steps:
            print('No automatic changes suggested. You can still add a custom transformation.')
    elif choice == 1:
        path = Path(ask('Recipe JSON path: ').strip('"').strip("'"))
        payload = json.loads(path.read_text(encoding='utf-8'))
        recipe = Recipe.from_dict(payload['recipe'] if payload.get('kind')=='preprocessing_proposal' else payload)
        validate_recipe(recipe,columns,require_approved=False)
    else:
        recipe = Recipe()
    recipe = _review(recipe,columns,choose,ask)
    service.save_recipe(recipe)
    return True


def _split_config(service, choose, ask):
    presplit = service.dataset.manifest.split_files
    fraction = .2
    if presplit:
        print('\nUsing the supplied split files. Test rows stay in the test set.')
        if 'validation' not in presplit:
            option = choose('Validation portion from the training upload', ['20%','15%','10%','Custom','None (keep all uploaded training rows)'])
            if option<3:
                fraction = [.2,.15,.1][option]
            elif option==4:
                fraction=0.
            else:
                while True:
                    try:
                        fraction = float(ask('Validation percentage (0 to less than 100): '))/100
                        if not 0<=fraction<1:
                            raise ValueError
                        break
                    except ValueError:
                        print('Enter a percentage at least 0 and less than 100.')
        ratios = (.7,.15,.15)
    else:
        option = choose('\nTrain / validation / test', ['70% / 15% / 15%', '80% / 10% / 10%',
                                                       '60% / 20% / 20%', '80% / 0% / 20%','Custom percentages'])
        if option<3:
            ratios = [(.7,.15,.15),(.8,.1,.1),(.6,.2,.2)][option]
        elif option==3:
            ratios=(.8,0.,.2)
        else:
            while True:
                try:
                    values = ask('Train, validation, test percentages (e.g. 75,15,10): ')
                    ratios = tuple(float(v.strip())/100 for v in values.split(','))
                    if len(ratios)!=3:
                        raise ValueError
                    SplitConfig(train=ratios[0],validation=ratios[1],test=ratios[2])
                    break
                except ValueError:
                    print('Enter three nonnegative percentages totaling 100; training must be positive.')
    default = 'stratified' if service.task().task_type=='classification' else 'random'
    choices = [f'{default.capitalize()} (usual choice)','Random','Keep groups together','Chronological']
    option = choose('Split strategy',choices)
    strategy = [default,'random','group','chronological'][option]
    extra = {}
    columns = [c for c in service.schema() if c!=service.task().target]
    if strategy in {'group','chronological'}:
        column = columns[choose('Split control column',columns)]
        # A control must not be transformed in the recipe; prepare_task validates this.
        extra['group_column' if strategy=='group' else 'time_column'] = column
        if strategy=='chronological':
            extra['time_format'] = ask('Time format (Enter for ISO8601): ') or 'ISO8601'
    while True:
        try:
            seed = int(ask('Random seed (Enter for 42): ') or '42')
            break
        except ValueError:
            print('Enter an integer seed.')
    return SplitConfig(strategy,*ratios,seed=seed,**extra), fraction


def show_artifacts(service):
    print('\nSaved project data')
    for label,path in service.artifacts():
        print(f'  {label}: {path}')
    state = service.state()
    if state.get('recipe'):
        print('  Recipe: '+str(service.dataset.directory/state['recipe']))
    if state.get('split'):
        print('  Prepared rows: '+', '.join(f'{k} {v:,}' for k,v in state['split']['prepared_counts'].items()))


def run(project, args, *, dataset_id=None, show_saved=True):
    from .project_cli import _choose as choose, _ask as ask
    service = ProjectPreparation(project,dataset_id,batch_size=args.batch_size)
    while True:
        state = service.state()
        if state.get('split'):
            if show_saved: show_artifacts(service)
            show_saved=True
            action = choose('\nPreparation complete', ['Finish','Create another split','Review a new recipe','Choose and train a model'])
            if action==3:
                from .model_wizard import run as model_run
                args.dataset_id=service.dataset.dataset_id
                return model_run(project,args)
            if action==0:
                print('Raw data, processed data, and splits are saved.')
                print('Next: '+command('workbench',project.name,'--models','--projects-dir',args.projects_dir))
                return 0
            if action==2:
                if not _recipe(service,choose,ask):
                    return 0
                continue
        elif not state.get('task'):
            if choose('\nNext: preprocessing', ['Review preprocessing now','Save for later'])==1:
                return 0
            _configure(service,choose,ask)
        if not service.state().get('recipe'):
            if not _recipe(service,choose,ask):
                return 0
        if not service.state().get('processed'):
            print('\nFitting the reviewed recipe for an inspection preview...')
            try:
                fitted,before,after = service.fit_preview()
            except (ValueError,TypeError) as exc:
                print(f'Preview needs a recipe change: {exc}')
                if choose('Next',['Review recipe again','Save for later'])==1:
                    return 0
                if not _recipe(service,choose,ask):
                    return 0
                continue
            print('\nBefore (up to 5 rows; original row index):')
            print(before.to_string(max_cols=8,max_colwidth=24,line_width=120))
            print('\nAfter (same input rows; removed rows are absent):')
            print(after.to_string(max_cols=8,max_colwidth=24,line_width=120))
            print('This inspection copy uses the source/training upload. Final split preprocessing will refit on training rows only.')
            action = choose('Save these changes?', ['Save processed data','Edit recipe','Save recipe and finish later'])
            if action==2:
                return 0
            if action==1:
                if not _recipe(service,choose,ask):
                    return 0
                continue
            processed = service.save_processed(fitted)
            print('Processed data saved: '+str(service.dataset.directory/processed['path']))
        next_step = choose('\nNext: data split',['Choose and save a split','Save for later','Review a new recipe'])
        if next_step==1:
            return 0
        if next_step==2:
            if not _recipe(service,choose,ask):
                return 0
            continue
        config, fraction = _split_config(service,choose,ask)
        print('Preparing splits and fitting transformations on training rows...')
        try:
            service.split(config,validation_fraction=fraction)
        except (ValueError,TypeError) as exc:
            print(f'Split could not be saved: {exc}')
            print('Your recipe and processed copy are still saved.')
            if choose('Next',['Choose split settings again','Finish for now'])==1:
                return 0
            continue
        show_artifacts(service)
        print('\nPreparation complete. Choose a model next.')
        from .model_wizard import run as model_run
        args.dataset_id=service.dataset.dataset_id
        return model_run(project,args)
