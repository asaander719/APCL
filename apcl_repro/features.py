"""Feature loading with explicit corruption errors; never replace missing data."""
from pathlib import Path
import zipfile

import numpy as np
import torch


def load_feature(path, trusted=False):
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f'Feature file not found: {p}. Check the selected YAML and feature overrides.')
    size = p.stat().st_size
    with p.open('rb') as stream:
        signature = stream.read(4)
    if signature == b'PK\x03\x04' and not zipfile.is_zipfile(p):
        raise ValueError(
            f'Incomplete or corrupt PyTorch archive: {p} ({size:,} bytes). '
            'ZIP central directory is missing. Restore the complete original file with the same '
            'item-index mapping. weights_only/trusted_pickle flags cannot restore missing bytes.')
    try:
        if p.suffix == '.npy':
            return torch.from_numpy(np.load(p, allow_pickle=False))
        return torch.load(p, map_location='cpu', weights_only=not trusted)
    except Exception as error:
        raise ValueError(
            f'Cannot load feature file {p} ({size:,} bytes): {error}. '
            'Verify file size and SHA256 against the original. Only for an intact, trusted '
            'legacy pickle use --trusted-feature-pickle (or experiment.trusted_pickle=true). '
            'Do not replace feature data with random/zero values.') from error


def load_tensor(path, trusted=False):
    value = load_feature(path, trusted)
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f'{path}: expected a finite two-dimensional indexed feature tensor')
    return value


def load_embedding(path, device, trusted=False):
    value = load_feature(path, trusted)
    if isinstance(value, dict):
        value = torch.stack([torch.as_tensor(v) for v in value.values()]).float()
        value = torch.cat([value, value.new_zeros((1, value.shape[1]))])
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f'{path}: expected a finite embedding matrix or ordered token dictionary')
    return value.float().to(device)
