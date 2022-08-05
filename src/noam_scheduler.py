class NoamScheduler:
    def __init__(self, optimizer, d_model, n_warmup_steps, scale=1.0):
        self._optimizer = optimizer
        self._d_model = d_model
        self._n_warmup_steps = n_warmup_steps
        self._n_steps = 0
        self._scale = scale

    def step(self):
        self._n_steps += 1
        lr = self._get_lr()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return self._scale * (self._d_model ** -0.5) * \
               min(self._n_steps ** (-0.5), self._n_steps * self._n_warmup_steps ** (-1.5))

    def state_dict(self):
        return dict(scale=self._scale, d_model=self._d_model, n_warmup_steps=self._n_warmup_steps,
                    n_steps=self._n_steps, optimizer=self._optimizer.state_dict())

    def load_state_dict(self, state_dict: dict):
        self._scale = state_dict['scale']
        self._d_model = state_dict['d_model']
        self._n_warmup_steps = state_dict['n_warmup_steps']
        self._n_steps = state_dict['n_steps']

        self._optimizer.load_state_dict(state_dict['_optimizer'])

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self._optimizer.param_groups]
