from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset


def _infinite(loader: DataLoader) -> Iterator:
    while True:
        yield from loader  # a new epoch re-shuffles


class MultiTaskLoader:
    """Yields {task_name: batch} each step. Each task cycles through its own data independently,
    so no task is truncated to the size of the smallest dataset."""

    def __init__(self, datasets: dict[str, Dataset], batch_size: int, num_workers: int = 0, seed: int = 0):
        self.loaders = {}
        for i, (name, ds) in enumerate(datasets.items()):
            g = torch.Generator()
            g.manual_seed(seed + i)
            self.loaders[name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                drop_last=len(ds) >= batch_size,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                generator=g,
            )
        self._iters = {name: _infinite(dl) for name, dl in self.loaders.items()}

    @property
    def task_names(self) -> list[str]:
        return list(self.loaders)

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, dict]:
        return {name: next(it) for name, it in self._iters.items()}
