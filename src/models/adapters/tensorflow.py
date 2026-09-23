import numpy as np
from .neural import NeuralAdapter


class TensorFlowAdapter(NeuralAdapter):
    def __init__(self,config,task_type):
        super().__init__(config,task_type)
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError('Install TensorFlow support: pip install -e ".[tensorflow]"') from exc
        self.tf = tf

    def _build(self):
        tf = self.tf
        tf.keras.utils.set_random_seed(self.config.seed)
        with tf.device('/CPU:0'):
            regularizer=None if self.options['regularizer']=='none' else (tf.keras.regularizers.L1 if self.options['regularizer']=='l1' else tf.keras.regularizers.L2)(self.options['regularization_strength'])
            self.model = tf.keras.Sequential([tf.keras.Input(shape=(self.architecture['inputs'],)),
                *[tf.keras.layers.Dense(n,activation=self.options['activation'],kernel_regularizer=regularizer) for n in self.architecture['hidden_sizes']],
                tf.keras.layers.Dense(self.architecture['outputs'],kernel_regularizer=regularizer)])
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) if self.task_type=='classification' else tf.keras.losses.Huber() if self.loss_name=='huber' else self.loss_name
            cls={'adam':tf.keras.optimizers.Adam,'adamw':tf.keras.optimizers.AdamW,'sgd':tf.keras.optimizers.SGD,'rmsprop':tf.keras.optimizers.RMSprop}[self.options['optimizer']]
            self.model.compile(optimizer=cls(self.options['learning_rate'],**({'weight_decay':0.0} if self.options['optimizer']=='adamw' else {})),loss=loss)

    def _train_batch(self,x,y):
        values = y if self.task_type=='classification' else np.asarray(y,dtype=np.float32).reshape(-1,1)
        with self.tf.device('/CPU:0'):
            self.model.train_on_batch(x,values)

    def _forward(self,x):
        with self.tf.device('/CPU:0'):
            return self.model(x,training=False).numpy()

    def _snapshot_best(self):
        self.best = [value.copy() for value in self.model.get_weights()]

    def _restore_best(self):
        self.model.set_weights(self.best)

    def _save_training(self,directory):
        self.model.save(directory/'checkpoint.keras')
        np.savez(directory/'best.npz',*self.best)

    def _load_training(self,directory):
        with self.tf.device('/CPU:0'):
            self.model = self.tf.keras.models.load_model(directory/'checkpoint.keras',safe_mode=True)
        with np.load(directory/'best.npz',allow_pickle=False) as arrays:
            self.best = [arrays[f'arr_{i}'].copy() for i in range(len(arrays.files))]

    def _save_model(self,directory):
        path = directory/'model.keras'
        self.model.save(path)
        return path

    def _load_model(self,directory):
        with self.tf.device('/CPU:0'):
            self.model = self.tf.keras.models.load_model(directory/'model.keras',compile=False,safe_mode=True)
