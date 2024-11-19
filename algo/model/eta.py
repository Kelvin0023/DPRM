import torch

class EtaFixed(torch.nn.Module):

    def __init__(
        self,
        base_eta=0.5,
        min_eta=0.1,
        max_eta=1.0,
        **kwargs,
    ):
        super().__init__()
        self.eta_logit = torch.nn.Parameter(torch.ones(1))
        self.min = min_eta
        self.max = max_eta

        # initialize such that eta = base_eta
        self.eta_logit.data = torch.atanh(
            torch.tensor([2 * (base_eta - min_eta) / (max_eta - min_eta) - 1])
        )

    def __call__(self, cond):
        """Match input batch size, but do not depend on input"""
        sample_data = cond["state"] if "state" in cond else cond["rgb"]
        B = len(sample_data)
        device = sample_data.device
        eta_normalized = torch.tanh(self.eta_logit)

        # map to min and max from [-1, 1]
        eta = 0.5 * (eta_normalized + 1) * (self.max - self.min) + self.min
        return torch.full((B, 1), eta.item()).to(device)