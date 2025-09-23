import torch
import itertools
import subprocess
import tempfile
import wandb

from pathlib import Path

from accelerate import Accelerator
from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from flax.nnx import Rngs

from smalldiffusion import (
    Unet, Scaled, ScheduleLogLinear, ScheduleSigmoid, samples, training_loop,
    MappedDataset, img_train_transform, img_normalize
)
from smalldiffusion.data import JaxRandomSampler, img_test_transform

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

    torch.save(model.state_dict(), f'checkpoints/checkpoint-{0:04}.pth')
    for i, ns in enumerate(training_loop(loader, model, train_schedule, epochs=epochs, lr=2e-4, accelerator=a)):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        wandb.log({'loss': ns.loss.item(), 'lr': ns.lr}, step=i)
        ema.update()
        if i % 1000 == 0 and checkpoint:
            with ema.average_parameters():
                torch.save(model.state_dict(), f'checkpoint-{i + 1:04}.pth')
                fid = eval_fid(TEST_DUMP, a, model, sample_schedule,
                        nfe=10, n=50_000,
                        batchsize=sample_batch_size)
                print(f"FID at step {i}: {fid}")
                wandb.log({'fid': fid}, step=i)

def dump_dataset(dataset, path, accelerator, batch_size=64):
    if not path.exists():
        print("Dumping test dataset...")
        path.mkdir(parents=True, exist_ok=True)
        counter = itertools.count()
        for i, img in zip(counter, dataset):
            save_image(img_normalize(img), path / f"{i:06}.png")

    saved_path = path.with_name(f"{path.name}.npz")
    if not saved_path.exists():
        save_fid_stats([str(path), str(saved_path)], batch_size, accelerator.device, dims=2048)
    return str(saved_path)

def eval_fid(dataset_path, accelerator, model, eval_schedule, nfe, n, batchsize=64):
    with tempfile.TemporaryDirectory() as out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model.eval()
        counter = itertools.count()
        for _ in tqdm(range(n // batchsize), desc=f"Evaluating FID"):
            *_, x0 = samples(model, eval_schedule.sample_sigmas(nfe), gam=2.1,
                        batchsize=batchsize, accelerator=accelerator)
            for i, img in zip(counter, x0):
                save_image(img_normalize(img), out_dir / f"{i:06}.png")
        model.train()
        return calculate_fid_given_paths((str(out_dir), str(dataset_path)), batchsize, accelerator.device, dims=2048)

if __name__=='__main__':
    main(checkpoint=True)
