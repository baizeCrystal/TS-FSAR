"""Render a few preselected cases with saved R/W, scores, and sampled frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video-root', type=Path, help='optional real frames, CPU decoding only')
    p.add_argument('--max-cases', type=int, default=5)
    a = p.parse_args()
    if a.output.exists() or a.max_cases < 1:
        p.error('new output directory and positive max-cases required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .core import read_rows, sample_indices
    episodes = {e['episode']: e for e in read_rows(a.run / 'episodes.jsonl')}
    choices = json.loads(a.cases.read_text())
    a.output.mkdir(parents=True)
    count = 0
    for condition, categories in choices.items():
        for category, case in categories.items():
            if count >= a.max_cases:
                return
            eid, qi, label = case['episode'], case['query_index'], case['label']
            base = np.load(a.run / f'e0_{eid:05d}.npz')
            changed = np.load(a.run / f'{condition}_{eid:05d}.npz')
            rival_values = changed['ts_dcm'][qi].copy()
            rival_values[label] = -np.inf
            rival = int(rival_values.argmax())
            fig, ax = plt.subplots(3, 3, figsize=(11, 9))
            for row, name in enumerate(('R true', 'R rival', 'W')):
                b = base['W'][qi, label] if row == 2 else base['R'][qi, [label, rival][row]]
                c = changed['W'][qi, label] if row == 2 else changed['R'][qi, [label, rival][row]]
                for col, (title, image) in enumerate((('E0', b), (condition, c), ('delta', c-b))):
                    lower, upper = min(b.min(), c.min()), max(b.max(), c.max())
                    if col == 2:
                        upper = max(float(np.abs(c-b).max()), 1e-8)
                        lower = -upper
                    im = ax[row, col].imshow(image, cmap='coolwarm', vmin=lower, vmax=upper)
                    ax[row, col].set_title(name + ' ' + title)
                    fig.colorbar(im, ax=ax[row, col], fraction=.045)
            fig.suptitle(f'{category}: episode {eid}, video {case["video_id"]}, true {label}, rival {rival}')
            fig.tight_layout()
            stem = f'{count:02d}_{condition}_{category}'
            fig.savefig(a.output / (stem + '_matrices.png'), dpi=130)
            plt.close(fig)
            if a.video_root:
                from .runtime import decode
                ep = episodes[eid]
                v = ep['query'][qi]
                meta = json.loads((a.run / f'{condition}_{eid:05d}.json').read_text())
                sampling = meta['sampling'][len(ep['support']) + qi]
                # Show a true/rival support too, with its actual current indices.
                rows = [('query E0', v, v['frame_indices']), ('query changed', v, sampling['indices'])]
                for cls, tag in ((label, 'true support'), (rival, 'rival support')):
                    j = next(i for i, s in enumerate(ep['support']) if s['episode_label'] == cls)
                    sv = ep['support'][j]
                    rows.append((tag, sv, meta['sampling'][j]['indices']))
                fig, ax = plt.subplots(len(rows), 8, figsize=(18, 2.3*len(rows)), squeeze=False)
                for r, (tag, video, indices) in enumerate(rows):
                    frames = decode(a.video_root, video, indices)
                    for col, frame in enumerate(frames):
                        ax[r, col].imshow(frame)
                        ax[r, col].set_title(f'{tag} f={indices[col]}', fontsize=8)
                        ax[r, col].axis('off')
                fig.tight_layout()
                fig.savefig(a.output / (stem + '_frames.png'), dpi=120)
                plt.close(fig)
            scores = {branch: {'e0': base[branch][qi].tolist(), 'changed': changed[branch][qi].tolist()}
                      for branch in ('ts_dcm', 'adapter_text', 'lsn_text', 'fusion')}
            (a.output / (stem + '_scores.json')).write_text(json.dumps(scores, indent=2))
            count += 1


if __name__ == '__main__':
    main()
