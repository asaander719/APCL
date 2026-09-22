"""Train the documented APCL example and select a checkpoint on validation only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .history import TrainingHistories
from .runtime import digest, evaluate_pairs, load_config, make_batch, setup, sync, write_json


def run(a):
    c = load_config(a.config)
    if a.seed is not None:
        c['seed'] = a.seed
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True)
    write_json(out/'config.json', c)
    write_json(out/'status.json', {'status': 'initializing'})
    model, arrays, source_rows, manifest = setup(c, a.data_root, a.device, a.trusted_feature_pickle)
    write_json(out/'manifest.json', manifest)
    device = torch.device(a.device)
    sampler = TrainingHistories(arrays['train'], c['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: c['lr_decay']**epoch)
    best, stale, best_epoch = -float('inf'), 0, 0
    batch_size = c['batch_size']
    for epoch in range(1, c['epochs']+1):
        sync(device); begin = time.perf_counter()
        model.train()
        histories = sampler.sample(arrays['train'], c['seed'], epoch)
        generator = torch.Generator().manual_seed(c['seed'] + epoch)
        order = torch.randperm(len(arrays['train']), generator=generator).numpy()
        lr = optimizer.param_groups[0]['lr']
        loss_sum, used = 0., 0
        for start in range(0, len(order)-batch_size+1, batch_size):
            idx = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            delta, *auxiliary = model(make_batch(arrays['train'][idx], histories[idx], device), train=True)
            loss = -F.logsigmoid(delta).sum() + c['model']['w_infoNCE'] * sum(auxiliary)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()); used += len(idx)
        if used == 0:
            raise ValueError('Not enough rows for one complete training batch')
        scheduler.step(); sync(device)
        train_s = time.perf_counter() - begin
        begin = time.perf_counter()
        auc, delta, _ = evaluate_pairs(model, arrays['valid'], arrays['valid_history'], device, c['evaluation_batch_size'])
        sync(device); validation_s = time.perf_counter() - begin
        if auc > best:
            best, best_epoch, stale = auc, epoch, 0
            torch.save(model.state_dict(), out/'best.pt.tmp')
            (out/'best.pt.tmp').replace(out/'best.pt')
            np.savez_compressed(out/'best_validation.npz', rows=arrays['valid'],
                                source_row=source_rows['valid'], delta=delta)
        else:
            stale += 1
        record = dict(epoch=epoch, lr=lr, loss_per_training_row=loss_sum/used,
                      validation_supplied_pair_auc=auc, best_validation_auc=best,
                      best_epoch=best_epoch, train_s=train_s, validation_s=validation_s)
        with (out/'epochs.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        write_json(out/'status.json', dict(status='training', **record))
        print(json.dumps(record), flush=True)
        if stale >= c['patience']:
            break
    result = dict(status='validation_complete', best_epoch=best_epoch, epochs_run=epoch,
                  validation_supplied_pair_auc=best, seed=c['seed'], n_valid=len(arrays['valid']),
                  checkpoint_sha256=digest(out/'best.pt'), test_evaluated=False)
    write_json(out/'result.json', result)
    write_json(out/'status.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='reproduction/configs/iqon_rb.json')
    p.add_argument('--data-root', required=True, help='Directory containing IQON3000/')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int)
    p.add_argument('--trusted-feature-pickle', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
