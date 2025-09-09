import torch
import click
import tqdm


import torch

from pathlib import Path
from torchvision.utils import save_image
from accelerate import Accelerator

from smalldiffusion import (
    Unet, Scaled, ScheduleLogLinear, ScheduleSigmoid, samples, training_loop,
    MappedDataset, img_train_transform, img_normalize
)

@click.option("--schedule-type", type=str, default="log_linear")
@click.option("--num_samples", type=int, default=50_000)
@click.option("--sample-batch-size", type=int, default=64)
@click.argument("out_dir")
@click.argument("model_path")
@click.command()
def main(model_path, out_dir, sample_batch_size, num_samples, schedule_type):
    out_dir = Path(out_dir)
    params = torch.load(model_path)
    a = Accelerator()
    model = Scaled(Unet)(32, 3, 3, ch=128, ch_mult=(1, 2, 2, 2), attn_resolutions=(16,))
    model.load_state_dict(params)
    model.to("cuda:0")

    num_batches = num_samples // sample_batch_size
    num = 0
    # Sample
    if schedule_type == "log_linear":
        sample_schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=35, N=1000)
    elif schedule_type == "sigmoid":
        sample_schedule = ScheduleSigmoid(N=1000)
    for i in tqdm.tqdm(range(num_batches)):
        *_, x0 = samples(model, sample_schedule.sample_sigmas(10), gam=2.1,
                          batchsize=sample_batch_size, accelerator=a)
        for img in x0:
            save_image(img_normalize(img), out_dir / f"{num:06}.png")
            num = num + 1

if __name__=="__main__":
    main()
