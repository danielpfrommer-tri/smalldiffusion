from types import SimpleNamespace
from typing import Tuple, Union, Optional
import torch
from tqdm import tqdm
import wandb

from pathlib import Path

from accelerate import Accelerator
from torch import nn
from torch.autograd.functional import jvp
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch_ema import ExponentialMovingAverage as EMA

from flax.nnx import Rngs

from smalldiffusion import (
    Unet, Scaled, ScheduleLogLinear, ScheduleSigmoid,
    MappedDataset, img_train_transform, img_normalize, ModelMixin, Schedule
)
from smalldiffusion.data import JaxRandomSampler, img_test_transform
from cifar_utils import dump_dataset, eval_fid
from smalldiffusion.model import SigmaEmbedderSinCos

class UnetST(Unet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau_embed = SigmaEmbedderSinCos(self.temb_ch)

    def forward(self, x, sigma, tau, cond=None):
        # Embeddings
        emb = self.sig_embed(x.shape[0], sigma.squeeze())
        emb += self.tau_embed(x.shape[0], tau.squeeze())
        return self.fwd_emb(x, emb, cond)

def generate_train_sample(x0: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                          schedule: Schedule, conditional: bool=False, frac_equal: float=0.5):
    cond = x0[1] if conditional else None
    init: torch.Tensor = x0[0] if conditional else x0 # type: ignore

    def get_noise():
        noise = schedule.sample_batch(init)
        while len(noise.shape) < len(init.shape):
            noise = noise.unsqueeze(-1)
        return noise

    sigma, tau = get_noise(), get_noise()
    sigma, tau = torch.min(sigma, tau), torch.max(sigma, tau)

    # random mask with same shape as sigma, with frac_equal of True
    mask = torch.rand(sigma.shape, device=sigma.device) < frac_equal
    sigma = torch.where(mask, tau, sigma)

    eps = torch.randn_like(init)
    return x0, sigma, tau, eps, cond

def training_loop(loader      : DataLoader,
                  model       : ModelMixin,
                  schedule    : Schedule,
                  accelerator : Optional[Accelerator] = None,
                  epochs      : int = 10000,
                  lr          : float = 1e-3,
                  frac_equal  : float = 0.5,
                  conditional : bool = False):
    accelerator = accelerator or Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # type: ignore
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    for _ in (pbar := tqdm(range(epochs))):
        for x0 in loader:
            model.train() # type: ignore
            optimizer.zero_grad()
            x0, sigma, tau, eps, cond = generate_train_sample(x0, schedule, conditional, frac_equal)
            xt = x0 + tau * eps
            v = eps - x0
            s0 = torch.full_like(sigma, schedule.sigmas[0])
            s1 = torch.full_like(sigma, schedule.sigmas[-1])
            u, dudt = jvp(model, (xt, sigma, tau), (v, s0, s1), create_graph=True)
            u_tgt = v - (tau - sigma) * dudt
            loss = nn.MSELoss()(u, u_tgt.detach())
            yield SimpleNamespace(**locals()) # For extracting training statistics
            accelerator.backward(loss)
            optimizer.step()

def main(train_batch_size=64, epochs=1000, sample_batch_size=64, checkpoint=False):
    wandb.init()
    test_dataset = MappedDataset(
        CIFAR10('datasets', train=True, download=True,
                transform=img_test_transform()),
        lambda x: x[0]
    )
    a = Accelerator()
    TEST_DUMP = dump_dataset(test_dataset, Path('datasets/cifar10-fid-test'), a)

    train_schedule = ScheduleSigmoid(N=1000)
    sample_schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=35, N=1000)

    rngs = Rngs(43)
    model = Scaled(Unet)(32, 3, 3, ch=128, ch_mult=(1, 2, 2, 2), attn_resolutions=(16,), rngs=rngs)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/checkpoint-initial.pth')

    dataset = MappedDataset(CIFAR10('datasets', train=True, download=True,
                                    transform=img_train_transform(rngs.flip)),
                            lambda x: x[0])
    loader = DataLoader(dataset, batch_size=train_batch_size, sampler=JaxRandomSampler(dataset, rngs.data))

    # Train
    ema = EMA(model.parameters(), decay=0.9999)
    ema.to(a.device)

    batches_per_epoch = len(loader.dataset) // train_batch_size + 1

    torch.save(model.state_dict(), f'checkpoints/checkpoint-{0:04}.pth')
    for i, ns in enumerate(training_loop(loader, model, train_schedule, epochs=epochs, lr=2e-4, accelerator=a)):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        wandb.log({'loss': ns.loss.item(), 'lr': ns.lr}, step=i)
        ema.update()
        if i % (batches_per_epoch * 30) == 0 and checkpoint and i > 0:
            with ema.average_parameters():
                torch.save(model.state_dict(), f'checkpoint-{i + 1:04}.pth')
                fid = eval_fid(TEST_DUMP, a, model, sample_schedule,
                        nfe=10, n=10_000,
                        batchsize=sample_batch_size)
                print(f"FID at step {i}: {fid}")
                wandb.log({'fid': fid}, step=i)

if __name__=='__main__':
    main(checkpoint=True)
