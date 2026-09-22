"""Self-contained data loading for the documented IQON3000 RB example."""
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .features import load_tensor, load_embedding
from .history import query_histories
from .models.APCL import APCL


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def make_batch(rows, histories, device):
    return [torch.as_tensor(rows[:, k], dtype=torch.long, device=device) for k in range(4)] + [
        torch.as_tensor(histories, dtype=torch.long, device=device),
        torch.ones(len(rows), device=device), torch.ones(len(rows), device=device)]


def load_config(path):
    c = json.loads(Path(path).read_text())
    if (c['dataset'], c['mode']) != ('IQON3000', 'RB'):
        raise ValueError('This minimal release documents IQON3000 RB only')
    if c['model']['include_indirect_score'] or c['model']['use_weighted_loss']:
        raise ValueError('This release evaluates the documented historical forward score')
    return c


def read_splits(paths):
    """Preserve within-split repeats; remove cross-split duplicate triples."""
    arrays, audit, source_rows = {}, {}, {}
    seen = set()
    for split in ('train', 'valid', 'test'):
        raw = np.loadtxt(paths[split], delimiter=',', ndmin=2)
        if raw.shape[1] != 4 or not np.isfinite(raw).all() or (raw < 0).any() or not np.equal(raw, raw.astype(np.int64)).all():
            raise ValueError(f'{split}: expected integer user,specified,positive,negative columns')
        rows = raw.astype(np.int64)
        if (rows[:, 2] == rows[:, 3]).any():
            raise ValueError(f'{split}: positive equals supplied negative')
        keep = np.array([tuple(row[:3]) not in seen for row in rows])
        arrays[split] = rows[keep]
        source_rows[split] = np.flatnonzero(keep) + 1
        seen.update(map(tuple, arrays[split][:, :3]))
        audit[split] = dict(original_n=len(rows), retained_n=int(keep.sum()),
                            removed_source_rows=(np.flatnonzero(~keep)+1).tolist(), sha256=digest(paths[split]))
    return arrays, source_rows, audit


def setup(c, data_root, device, trusted=False):
    root = Path(data_root).expanduser().resolve() / 'IQON3000'
    paths = {s: root/'data'/f'{s}_indexed.csv' for s in ('train', 'valid', 'test')}
    paths.update(user_map=root/'data/user_map.json', item_map=root/'data/item_map.json',
                 visual=root/'feat/visualfeatures_indexedtenseor',
                 text=root/'feat/textfeatures_indexedtenseor', embedding=root/'feat/smallnwjc2vec')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.cuda.manual_seed_all(c['seed'])
    torch.set_num_threads(c['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    m = dict(c['model'], dataset=c['dataset'], mode=c['mode'], device=device)
    m['user_num'] = len(json.loads(paths['user_map'].read_text()))
    m['item_num'] = len(json.loads(paths['item_map'].read_text()))
    visual = load_tensor(paths['visual'], trusted).float()
    text = load_tensor(paths['text'], trusted)
    embedding = load_embedding(paths['embedding'], 'cpu', trusted)
    if visual.shape != (m['item_num'], m['visual_feature_dim']):
        raise ValueError('Visual feature shape does not match item map')
    if text.shape != (m['item_num'], m['max_sentence']) or not torch.equal(text, text.long().to(text.dtype)):
        raise ValueError('Expected one indexed token sequence per item')
    text = text.long()
    if text.min() < 0 or text.max() >= len(embedding):
        raise ValueError('Text token ID outside embedding vocabulary')
    arrays, source_rows, audit = read_splits(paths)
    for split in ('train', 'valid', 'test'):
        rows = arrays[split]
        if not len(rows) or rows[:, 0].max() >= m['user_num'] or rows[:, 1:].max() >= m['item_num']:
            raise ValueError('Split indices do not match maps')
        arrays[split+'_history'] = query_histories(arrays['train'], rows, m['u_pb_num'],
                                                  m['top_u'], c['history_seed'])
    model = APCL(SimpleNamespace(**m), embedding, visual, text).to(device)
    sources = {str(p.relative_to(Path(__file__).parent)): digest(p)
               for p in sorted(Path(__file__).parent.rglob('*.py'))}
    manifest = dict(data={name: digest(p) for name, p in paths.items()}, split_audit=audit,
                    code=sources, torch=str(torch.__version__), numpy=str(np.__version__),
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    n_users=m['user_num'], n_items=m['item_num'])
    return model, arrays, source_rows, manifest


@torch.no_grad()
def evaluate_pairs(model, rows, histories, device, batch_size):
    """AUC over supplied pairs, matching the validation selection criterion."""
    model.eval()
    previous = model.CL
    model.CL = False
    scores = []
    try:
        for start in range(0, len(rows), batch_size):
            sl = slice(start, start+batch_size)
            scores.append(model(make_batch(rows[sl], histories[sl], device), train=False)[0].cpu().numpy())
    finally:
        model.CL = previous
    delta = np.concatenate(scores)
    if not np.isfinite(delta).all():
        raise FloatingPointError('Nonfinite evaluation score')
    per_query = (delta > 0).astype(float) + .5 * (delta == 0)
    return float(per_query.mean()), delta, per_query
