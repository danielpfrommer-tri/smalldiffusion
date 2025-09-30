import torch
import typing as tp
import csv
import numpy as np
import jax

from torch.utils.data import Dataset, Sampler
from torchvision import transforms as tf
import torchvision.transforms.functional as tf_F

from typing import Iterator, Sized, Optional
from .model import RngStream

## Dataset utility functions

# Mainly used to discard labels and only output data
class MappedDataset(Dataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.fn(self.dataset[i])

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, rng: RngStream, p=0.5):
        super().__init__()
        self.rng = rng
        self.p = p

    def forward(self, img):
        if jax.random.bernoulli(self.rng(), self.p):
            return tf_F.hflip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"

def img_test_transform():
    return tf.Compose([
        tf.ToTensor(),
        tf.Lambda(lambda t: (t * 2) - 1)
    ])

def img_train_transform(rng: RngStream):
    return tf.Compose([
        RandomHorizontalFlip(rng),
        tf.ToTensor(),
        tf.Lambda(lambda t: (t * 2) - 1)
    ])

img_normalize = lambda x: ((x + 1)/2).clamp(0, 1)

class JaxRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized
    replacement: bool

    def __init__(
        self,
        data_source: Sized,
        rng_stream: RngStream,
        replacement: bool = False,
        num_samples: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.rng_stream = rng_stream

        if not isinstance(self.replacement, bool):
            raise TypeError(
                f"replacement should be a boolean value, but got replacement={self.replacement}"
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
            )

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from (int(i) for i in self.rng_stream.randint((32,), 0, n))
        else:
            assert jax is not None, "JAX is required for RandomSampler without replacement"
            for _ in range(self.num_samples // n):
                yield from (int(i) for i in jax.random.permutation(self.rng_stream(), n))
            if self.num_samples % n != 0:
                yield from (int(i) for i in jax.random.choice(self.rng_stream(), n, (self.num_samples % n,), replace=False))

    def __len__(self) -> int:
        return self.num_samples

## Toy datasets

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N, center=(0,0), scale=1.0):
        t = tmin + torch.linspace(0, 1, N) * tmax
        center = torch.tensor(center).unsqueeze(0)
        self.vals = center + scale * torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]

class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=15, delimiter='\t', scale=50, offset=50):
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        return self.points[i % len(self.points)]

def interpolate_polyline(points, num_samples):
    """
    Given a list of 2D points defining a polyline,
    sample num_samples points uniformly along its arc length.
    """
    points = np.array(points)
    # Compute distances between consecutive points
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumdist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumdist[-1]
    # Equally spaced arc-length values
    sample_dists = np.linspace(0, total_length, num_samples)
    samples = []
    for d in sample_dists:
        # Find which segment d falls in
        seg = np.searchsorted(cumdist, d, side='right') - 1
        seg = min(seg, len(dists) - 1)
        # Compute local interpolation parameter
        t = (d - cumdist[seg]) / dists[seg] if dists[seg] > 0 else 0
        sample = (1 - t) * points[seg] + t * points[seg + 1]
        samples.append(sample)
    return np.array(samples)

class TreeDataset(Dataset):
    def __init__(self, branching_factor=4, depth=3, num_samples_per_path=30):
        """
        Initializes a tree dataset where each leaf of the tree lies on the
        circle of radius 1. The tree is constructed with the given branching_factor
        and depth. Each leaf’s path is sampled uniformly, and each sampled point
        is given the label of the leaf.

        Parameters:
         - branching_factor (int): number of branches at each node.
         - depth (int): number of branchings (excluding the root).
                        Total leaves = branching_factor ** depth.
         - num_samples_per_path (int): number of points sampled along each path.
        """
        self.data = []
        self.total_leaves = branching_factor ** depth

        # Iterate over each leaf index
        for i in range(self.total_leaves):
            # Build the sequence of nodes along the path from the root to this leaf.
            # Start with the root at (0, 0)
            path_points = [np.array([0.0, 0.0])]

            # For each level l (1 to depth), compute the branch node.
            for l in range(1, depth + 1):
                # Group size for this level (leaves per branch node)
                group_size = branching_factor ** (depth - l)  # For l == depth, group_size == 1.
                # A_l is the branch index for level l
                A_l = i // group_size
                # Compute the average index for all leaves under this branch node
                avg_index = A_l * group_size + (group_size - 1) / 2.0
                # Compute angular coordinate (all leaves are uniformly spaced on the circle)
                theta = avg_index * (2 * np.pi / self.total_leaves)
                # Set radius proportional to the level (leaf at level==depth has r==1)
                r = l / depth
                p = np.array([r * np.cos(theta), r * np.sin(theta)])
                path_points.append(p)

            # Sample points uniformly along the polyline defined by the path
            samples = interpolate_polyline(path_points, num_samples_per_path)
            # Append each sample with its label (the leaf index)
            for sample in samples:
                # Each item is a tuple: (2D coordinate tensor, label)
                self.data.append((torch.tensor(sample, dtype=torch.float32), i))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
