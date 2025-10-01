import torch
import wandb

from pathlib import Path

from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch_ema import ExponentialMovingAverage as EMA

from flax.nnx import Rngs

from smalldiffusion import (
    Unet, Scaled, ScheduleLogLinear, ScheduleSigmoid, samples, training_loop,
    MappedDataset, img_train_transform, img_normalize
)
from smalldiffusion.data import JaxRandomSampler, img_test_transform
from cifar_utils import dump_dataset, eval_fid

def main(train_batch_size=256, epochs=1000, sample_batch_size=64, checkpoint=False):
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
        if i % (batches_per_epoch * 30) == 0 and checkpoint:
            with ema.average_parameters():
                torch.save(model.state_dict(), f'checkpoint-{i + 1:04}.pth')
                fid = eval_fid(TEST_DUMP, a, model, sample_schedule,
                        nfe=10, n=50_000,
                        batchsize=sample_batch_size)
                print(f"FID at step {i}: {fid}")
                wandb.log({'fid': fid}, step=i)

if __name__=='__main__':
    main(checkpoint=True)
