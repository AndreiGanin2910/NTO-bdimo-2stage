import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
import zipfile

zip_path = "D_data/NTO_BDML_2G_hold_out_validation.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

DATA_PATH = 'D_data/csv/'

files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
data = {f: pd.read_csv(DATA_PATH + f) for f in files}

sigs = {}
for fname, df in data.items():
    v = df.values.astype(np.float64)
    nr, nc = v.shape
    x = v.flatten()
    q = np.linspace(0, 1, 61)
    
    gs = np.array([np.mean(x), np.std(x), np.median(x), np.min(x), np.max(x),
                   np.max(x) - np.min(x), np.quantile(x, 0.75) - np.quantile(x, 0.25),
                   np.quantile(x, 0.01), np.quantile(x, 0.05), np.quantile(x, 0.95), np.quantile(x, 0.99)])
    
    h, _ = np.histogram(x, bins=48, density=True)
    h = h / (h.sum() + 1e-12)
    
    rm = np.mean(v, axis=1)
    rs_arr = np.std(v, axis=1)
    rmed = np.median(v, axis=1)
    rr = np.max(v, axis=1) - np.min(v, axis=1)
    rs = np.concatenate([np.quantile(rm, q), np.quantile(rs_arr, q), np.quantile(rmed, q), np.quantile(rr, q)])
    
    cf = []
    for i in range(nc):
        c = v[:, i]
        sk = pd.Series(c).skew()
        ku = pd.Series(c).kurtosis()
        f = np.array([np.mean(c), np.std(c), np.median(c), np.min(c), np.max(c),
                      np.quantile(c, 0.05), np.quantile(c, 0.10), np.quantile(c, 0.25),
                      np.quantile(c, 0.75), np.quantile(c, 0.90), np.quantile(c, 0.95),
                      np.quantile(c, 0.75) - np.quantile(c, 0.25), np.max(c) - np.min(c),
                      sk if not np.isnan(sk) else 0.0, ku if not np.isnan(ku) else 0.0])
        f = np.concatenate([f, np.quantile(c, q)])
        cf.append(((np.mean(c), np.std(c), np.median(c)), f))
    cf.sort(key=lambda t: t[0])
    cs = np.concatenate([t[1] for t in cf])
    
    pcorr = np.corrcoef(v.T)
    pcorr = np.nan_to_num(pcorr, nan=0.0, posinf=0.0, neginf=0.0)
    pe = np.sort(np.linalg.eigvalsh(pcorr))
    pp = np.array([pcorr[i, j] for i in range(nc) for j in range(i + 1, nc)])
    pp = np.sort(pp)
    if len(pp) > 0:
        ps = np.array([np.mean(pp), np.std(pp), np.median(pp), np.min(pp), np.max(pp),
                       np.quantile(pp, 0.25), np.quantile(pp, 0.75), np.mean(np.abs(pp))])
    else:
        ps = np.zeros(8)
    
    scorr = np.zeros((nc, nc))
    for i in range(nc):
        scorr[i, i] = 1.0
        for j in range(i + 1, nc):
            r, _ = spearmanr(v[:, i], v[:, j])
            r = 0.0 if np.isnan(r) else r
            scorr[i, j] = scorr[j, i] = r
    se = np.sort(np.linalg.eigvalsh(scorr))
    sp = np.array([scorr[i, j] for i in range(nc) for j in range(i + 1, nc)])
    sp = np.sort(sp)
    if len(sp) > 0:
        ss = np.array([np.mean(sp), np.std(sp), np.median(sp), np.min(sp), np.max(sp), np.mean(np.abs(sp))])
    else:
        ss = np.zeros(6)
    
    sigs[fname] = {'nc': nc, 'gs': gs, 'h': h, 'rs': rs, 'cs': cs,
                   'pe': pe, 'pp': pp, 'ps': ps, 'se': se, 'sp': sp, 'ss': ss}

groups = defaultdict(list)
for f, s in sigs.items():
    groups[s['nc']].append(f)

pairs = []
for nc, fs in groups.items():
    if len(fs) < 2:
        continue
    
    n = len(fs)
    dm = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = sigs[fs[i]], sigs[fs[j]]
            
            t = 0.0
            for k, w, use_l2 in [('pe', 17.0, True), ('se', 14.5, True), ('pp', 10.0, False),
                                  ('sp', 7.0, False), ('ps', 6.0, False), ('ss', 5.0, False),
                                  ('cs', 4.0, False), ('rs', 3.5, False), ('gs', 2.5, False), ('h', 2.0, False)]:
                a = np.nan_to_num(s1[k].flatten())
                b = np.nan_to_num(s2[k].flatten())
                if len(a) != len(b):
                    t += w * 1e8
                elif use_l2:
                    sc = (np.linalg.norm(a) + np.linalg.norm(b)) / 2.0 + 1e-12
                    t += w * np.linalg.norm(a - b) / sc
                else:
                    sc = (np.mean(np.abs(a)) + np.mean(np.abs(b))) / 2.0 + 1e-12
                    t += w * np.mean(np.abs(a - b)) / sc
            
            dm[i, j] = dm[j, i] = t
    
    edges = sorted([(dm[i, j], i, j) for i in range(n) for j in range(i + 1, n)])
    
    used = set()
    for _, i, j in edges:
        if i not in used and j not in used:
            used.add(i)
            used.add(j)
            pairs.append(sorted([fs[i].replace('.csv', ''), fs[j].replace('.csv', '')]))

result = pd.DataFrame(pairs, columns=['part1', 'part2'])
result = result.sort_values('part1').reset_index(drop=True)
result.to_csv('D_outputs/final_submission.csv', index=False)