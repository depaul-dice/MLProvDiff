class TransformerScheduler:
    def __init__(self, optimizer, warmup_steps, d_model):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_steps = 0

    def step(self):
        self.current_steps += 1
        lr = self.d_model ** (-0.5) * min(self.current_steps ** (-0.5), self.current_steps * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
