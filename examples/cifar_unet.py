import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
from tqdm import tqdm

from smalldiffusion import (
    Unet, Scaled, ScheduleLogLinear, ScheduleSigmoid, samples, training_loop,
    MappedDataset, img_train_transform, img_normalize
)

def main(train_batch_size=256, epochs=1000, sample_batch_size=64):
    # Setup
    a = Accelerator()
    train_schedule = ScheduleSigmoid(N=1000)
    model = Scaled(Unet)(32, 3, 3, ch=128, ch_mult=(1, 2, 2, 2), attn_resolutions=(16,))
    torch.save(model.state_dict(), 'checkpoint-initial.pth')
    dataset = MappedDataset(CIFAR10('datasets', train=True, download=True,
                                    transform=img_train_transform),
                            lambda x: x[0])
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    # Train
    ema = EMA(model.parameters(), decay=0.9999)
    ema.to(a.device)
    for i, ns in enumerate(training_loop(loader, model, train_schedule, epochs=epochs, lr=2e-4, accelerator=a)):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()
        if i % 1000 == 0:
            with ema.average_parameters():
                torch.save(model.state_dict(), f'checkpoint-{i:04}.pth')


    # Sample
    sample_schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=35, N=1000)
    with ema.average_parameters():
        *xt, x0 = samples(model, sample_schedule.sample_sigmas(10), gam=2.1,
                          batchsize=sample_batch_size, accelerator=a)
        save_image(img_normalize(make_grid(x0)), 'samples.png')
        torch.save(model.state_dict(), 'checkpoint-final.pth')

if __name__=='__main__':
    main()
