"""Evaluate a validation-selected checkpoint once and save auditable pair scores."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .runtime import digest, evaluate_pairs, load_config, setup, sync, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='reproduction/configs/iqon_rb.json')
    p.add_argument('--data-root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--split', choices=['valid', 'test'], default='test')
    p.add_argument('--trusted-feature-pickle', action='store_true')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(f'Preserve previous evaluations: {out}')
    c = load_config(a.config)
    model, arrays, source_rows, manifest = setup(c, a.data_root, a.device, a.trusted_feature_pickle)
    state = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    model.load_state_dict(state)
    out.mkdir(parents=True)
    sync(a.device); begin = time.perf_counter()
    auc, delta, per_query = evaluate_pairs(model, arrays[a.split], arrays[a.split+'_history'],
                                          a.device, c['evaluation_batch_size'])
    sync(a.device); elapsed = time.perf_counter()-begin
    np.savez_compressed(out/'predictions.npz', rows=arrays[a.split], source_row=source_rows[a.split],
                        delta=delta, AUC=per_query)
    with (out/'per_query.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['source_row','user_id','specified_item_id','positive_item_id','negative_item_id','score_difference','AUC'])
        for index, row in enumerate(arrays[a.split]):
            w.writerow([source_rows[a.split][index], *row, float(delta[index]), float(per_query[index])])
    result = dict(dataset=c['dataset'], mode=c['mode'], split=a.split, n=len(delta), seed=c['seed'],
                  supplied_pair_auc=auc, ties=int((delta == 0).sum()), batch_size=c['evaluation_batch_size'],
                  evaluation_s=elapsed, checkpoint_sha256=digest(a.checkpoint),
                  protocol='Supplied positive/negative pairs; forward score; ties half; no rank metrics inferred',
                  data=manifest['data'], environment={k: manifest[k] for k in ['torch','numpy','cuda','gpu']})
    write_json(out/'result.json', result)
    write_json(out/'manifest.json', manifest)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
