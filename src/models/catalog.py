"""Dependency-light descriptions used by the CLI and future UI."""
from dataclasses import dataclass, field, asdict
from importlib.util import find_spec


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    name: str
    tasks: tuple
    modality: str
    description: str
    defaults: dict = field(default_factory=dict)
    extra: str = ''
    limitations: str = ''

    def to_dict(self):
        return asdict(self)


CATALOG = {}


def _add(key,name,tasks,description,defaults=None,modality='tabular',extra='',limitations=''):
    CATALOG[key] = CatalogEntry(key,name,tuple(tasks.split(',')),modality,description,defaults or {},extra,limitations)


_both='classification,regression'
for key,name,tasks,desc,params in [
    ('dummy','Baseline',_both,'Predicts a constant; measures whether learning adds value.',{}),
    ('linear','Linear baseline',_both,'Logistic classification or ridge regression; fast starting point.',{}),
    ('linear_regression','Linear regression','regression','Interpretable additive effects; sensitive to outliers.',{}),
    ('ridge','Ridge regression','regression','L2 regularization stabilizes correlated numeric features.',{'alpha':1.0}),
    ('lasso','Lasso regression','regression','L1 regularization can set coefficients to zero.',{'alpha':1.0}),
    ('elastic_net','Elastic net','regression','Combines L1 and L2 penalties.',{'alpha':1.0,'l1_ratio':0.5}),
    ('logistic_regression','Logistic regression','classification','Linear class boundaries and probabilities.',{'C':1.0,'max_iter':1000}),
    ('multinomial_logistic','Multinomial logistic regression','classification','Joint softmax classification for multiple classes.',{'C':1.0,'solver':'lbfgs','max_iter':1000}),
    ('decision_tree','Decision tree',_both,'Readable nonlinear rules; restrict depth to reduce overfitting.',{'max_depth':6,'min_samples_leaf':2}),
    ('random_forest','Random forest',_both,'Robust nonlinear tabular baseline using bagged trees.',{'n_estimators':100,'max_depth':None,'min_samples_leaf':1}),
    ('extra_trees','Extra trees',_both,'Randomized tree ensemble; useful alongside random forests.',{'n_estimators':100}),
    ('gradient_boosting','Gradient boosting',_both,'Sequential trees for nonlinear tabular relationships.',{'n_estimators':100,'learning_rate':0.1,'max_depth':3}),
    ('adaboost','AdaBoost',_both,'Boosts weak trees; may be sensitive to noisy observations.',{'n_estimators':50,'learning_rate':1.0}),
    ('svm','Support vector machine',_both,'Kernel boundaries on small/medium datasets; scales numeric inputs.',{'C':1.0,'kernel':'rbf'}),
    ('knn','K-nearest neighbors',_both,'Local similarity baseline; prediction can be costly on large data.',{'n_neighbors':5,'weights':'uniform'}),
    ('gaussian_nb','Gaussian Naive Bayes','classification','Fast classifier with per-class Gaussian feature assumptions.',{'var_smoothing':1e-9}),
    ('multinomial_nb','Multinomial Naive Bayes','classification','For nonnegative count-like features; negative inputs are invalid.',{'alpha':1.0}),
    ('bernoulli_nb','Bernoulli Naive Bayes','classification','For binary feature presence; binarizes numeric inputs at zero.',{'alpha':1.0,'binarize':0.0}),
    ('voting','Voting ensemble',_both,'Combines a linear model and random forest; soft probabilities for classification.',{}),
    ('stacking','Stacking ensemble',_both,'Learns to combine linear and forest predictions using internal cross-validation.',{'cv':3}),
]:
    _add('sklearn:'+key,name,tasks,desc,params)
for backend,name,params in [
    ('xgboost','XGBoost',{'n_estimators':100,'max_depth':6,'learning_rate':0.1}),
    ('lightgbm','LightGBM',{'n_estimators':100,'num_leaves':31,'learning_rate':0.1}),
    ('catboost','CatBoost',{'iterations':100,'depth':6,'learning_rate':0.1})]:
    _add(backend+':boosted_trees',name,_both,'Boosted trees for tabular prediction.',params,extra=backend,
         limitations='Uses the common train-fitted numeric/one-hot encoder; native categorical handling is not enabled.')
NEURAL_DEFAULTS={'epochs':30,'batch_size':64,'learning_rate':0.001,'optimizer':'adam',
                 'regularizer':'none','regularization_strength':0.0001,'loss':'auto'}
for backend in ('pytorch','tensorflow'):
    _add(backend+':mlp',backend.title()+' MLP',_both,'Dense neural network for tabular features; tune against tree baselines.',
         {**NEURAL_DEFAULTS,'hidden_sizes':[64,32],'patience':5},extra=backend)
for key,name,task,desc,params in [
    ('kmeans','K-means','clustering','Compact spherical clusters in scaled feature space.',{'n_clusters':8,'n_init':10}),
    ('hierarchical','Hierarchical clustering','clustering','Agglomerative cluster hierarchy; batch labels only.',{'n_clusters':2,'linkage':'ward'}),
    ('gaussian_mixture','Gaussian mixture','clustering','Soft probabilistic clusters with elliptical shapes.',{'n_components':2,'covariance_type':'full'}),
    ('dbscan','DBSCAN','clustering','Density clusters and noise points; tune eps after scaling.',{'eps':0.5,'min_samples':5}),
    ('pca','PCA','projection','Linear low-dimensional representation; variance ratios are reported.',{'n_components':2}),
    ('tsne','t-SNE','projection','Nonlinear exploratory embedding; distances between clusters are not global geometry.',{'n_components':2,'perplexity':30.0}),
    ('isolation_forest','Isolation Forest','anomaly','Unsupervised outlier ranking; -1 means flagged, not a confirmed anomaly.',{'n_estimators':100,'contamination':'auto'})]:
    _add('sklearn:'+key,name,task,desc,params,
         limitations='No out-of-sample prediction; export training assignments/embedding.' if key in {'hierarchical','dbscan','tsne'} else '')
for key,name in [('lenet','LeNet-style CNN'),('alexnet','AlexNet'),('vgg16','VGG16'),('resnet18','ResNet18'),('vit_b_16','Vision Transformer')]:
    _add('torchvision:'+key,name,'classification','Image classification with lazy image decoding and training-only labels.',
         {**NEURAL_DEFAULTS,'epochs':10,'batch_size':16,'learning_rate':0.0001,'pretrained':False},'image','vision',
         'Default weights are random. pretrained=true downloads torchvision weights (except LeNet). AlexNet/VGG/ViT require substantial RAM/compute.')
for key in ('lstm','gru'):
    _add('pytorch:'+key,key.upper(),'forecasting','One-step univariate forecasting with chronological windows.',
         {**NEURAL_DEFAULTS,'epochs':20,'window':24,'hidden_size':64,'num_layers':1},'time_series','pytorch',
         'Validation is rolling one-step forecasting using observed history; not a multi-step forecast score.')
_add('statsmodels:arima','ARIMA','forecasting','Univariate forecasting for regular time intervals.',
     {'order':[1,1,1]},'time_series','timeseries','Needs a strictly ordered, regular time index. No seasonal/exogenous terms in this adapter.')
for key,name,task,checkpoint in [('bert','BERT','classification','google-bert/bert-base-uncased'),
                               ('gpt','GPT-2','language_modeling','openai-community/gpt2'),
                               ('t5','T5','text_to_text','google-t5/t5-small')]:
    _add('transformers:'+key,name,task,'Fine-tunes a pretrained Hugging Face transformer with saved tokenizer.',
         {**NEURAL_DEFAULTS,'epochs':3,'batch_size':8,'learning_rate':0.00002,'optimizer':'adamw',
          'checkpoint':checkpoint,'revision':'main','max_length':128},'text','text',
         'Downloads model/tokenizer files; pin revision to a commit for repeatable downloads. No remote Python code is enabled. GPT is the open GPT-2 architecture, not hosted OpenAI GPT services.')
for key,name in [('svd','Truncated SVD'),('als','Implicit ALS')]:
    _add(('scipy:' if key=='svd' else 'implicit:')+key,name,'recommendation',
         'Factorizes a sparse positive user–item interaction matrix for personalized ranking.',
         {'factors':32,**({'iterations':20,'regularization':0.01} if key=='als' else {})},'recommendation',
         'recommendation' if key=='als' else '',
         'Implicit feedback ranking, not explicit star-rating regression; unknown users/items need a fallback. Duplicate user–item pairs are aggregated before splitting.')


def suggest(task,modality='tabular'):
    preferred={'classification':['sklearn:logistic_regression','sklearn:random_forest','xgboost:boosted_trees'],
               'regression':['sklearn:ridge','sklearn:random_forest','xgboost:boosted_trees'],
               'clustering':['sklearn:kmeans','sklearn:gaussian_mixture'],
               'projection':['sklearn:pca','sklearn:tsne'],'anomaly':['sklearn:isolation_forest']}
    return [e for e in CATALOG.values() if e.modality==modality and task in e.tasks
            and (modality!='tabular' or e.key in preferred.get(task,[]))]


def describe(key):
    if key not in CATALOG:
        raise ValueError('Unknown model; run workbench models list')
    e=CATALOG[key]
    return '\n'.join([f'{e.name} ({e.key})',f'Input: {e.modality}; tasks: {", ".join(e.tasks)}',e.description,
        f'Defaults: {e.defaults}',f'Install: python -m pip install -e ".[{e.extra}]"' if e.extra else 'Included in the base installation.',e.limitations])
