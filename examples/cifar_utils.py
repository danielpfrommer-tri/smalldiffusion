import torch
import itertools
import tempfile
from pathlib import Path

from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats
from torchvision.utils import save_image
from tqdm import tqdm

from smalldiffusion import samples, img_normalize


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