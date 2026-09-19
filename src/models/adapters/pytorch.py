from .neural import NeuralAdapter


class TorchAdapter(NeuralAdapter):
    def __init__(self,config,task_type):
        super().__init__(config,task_type)
        try:
            import torch
        except ImportError as exc:
            raise ImportError('Install PyTorch support: pip install -e ".[pytorch]"') from exc
        self.torch = torch

    def _build(self):
        torch = self.torch
        torch.manual_seed(self.config.seed)
        dims = [self.architecture['inputs'],*self.architecture['hidden_sizes'],self.architecture['outputs']]
        layers = []
        for index,(a,b) in enumerate(zip(dims,dims[1:])):
            layers.append(torch.nn.Linear(a,b))
            if index<len(dims)-2:
                layers.append(torch.nn.ReLU())
        self.model = torch.nn.Sequential(*layers).cpu()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.options['learning_rate'])

    def _train_batch(self,x,y):
        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(torch.from_numpy(x))
        if self.task_type=='classification':
            loss = torch.nn.functional.cross_entropy(output,torch.as_tensor(y,dtype=torch.long))
        else:
            loss = torch.nn.functional.mse_loss(output.reshape(-1),torch.as_tensor(y,dtype=torch.float32))
        loss.backward()
        self.optimizer.step()

    def _forward(self,x):
        self.model.eval()
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x)).numpy()

    def _snapshot_best(self):
        self.best = {k:v.detach().clone() for k,v in self.model.state_dict().items()}

    def _restore_best(self):
        self.model.load_state_dict(self.best)

    def _save_training(self,directory):
        self.torch.save({'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                         'best':self.best,'rng':self.torch.get_rng_state()},directory/'checkpoint.pt')

    def _load_training(self,directory):
        self._build()
        state = self.torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=True)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.best = state['best']
        self.torch.set_rng_state(state['rng'])

    def _save_model(self,directory):
        path = directory/'model.pt'
        self.torch.save(self.model.state_dict(),path)
        return path

    def _load_model(self,directory):
        self._build()
        self.model.load_state_dict(self.torch.load(directory/'model.pt',map_location='cpu',weights_only=True))
