import json, re, sys
from pathlib import Path
wd = Path('results/stage3/_judge_workdir')
cell = sys.argv[1] if len(sys.argv) > 1 else 'cuad__v4t-corpus-tuned__calibration__seed42'
nn = sys.argv[2] if len(sys.argv) > 2 else '00'
which = sys.argv[3] if len(sys.argv) > 3 else 'nz'  # nz | all | top


def parse(nn):
    txt = (wd / f'rejudge__{cell}__part{nn}.txt').read_text('utf-8').splitlines()
    out = {}; cur = None
    for L in txt:
        m = re.match(r'\[(\d+)\] qid=(\S+)', L)
        if m:
            cur = m.group(2); out[cur] = {'q': '', 'gold': '', 'pred': ''}
        elif cur:
            if L.startswith('  EXPECT='): out[cur]['q'] = L.split('| Q: ', 1)[-1]
            elif L.startswith('  GOLD: '): out[cur]['gold'] = L[8:]
            elif L.startswith('  PRED: '): out[cur]['pred'] = L[8:]
    return out


src = parse(nn)
sc = json.loads((wd / f'rejudge_scores__{cell}__part{nn}.json').read_text('utf-8'))
from collections import Counter
dist = Counter(v[0] for v in sc.values())
nz = [(q, v) for q, v in sc.items() if v[0] > 0]
print(f'{cell} part{nn}: {len(sc)} entries, dist={dict(sorted(dist.items()))}')
sel = nz if which == 'nz' else list(sc.items())
hi = [(q, v) for q, v in nz if v[0] >= 0.75]
show = (hi[:6] + nz[:6]) if which == 'top' else sel[:12]
for q, (s, r) in show:
    e = src.get(q, {})
    print('=' * 78)
    print(f'SCORE={s}  ...{q[-36:]}')
    print('Q   :', e.get('q', '')[:88])
    print('GOLD:', e.get('gold', '')[:130])
    print('PRED:', e.get('pred', '')[:165])
    print('RAT :', r[:150])
