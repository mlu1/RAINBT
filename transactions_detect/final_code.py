import os, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
try:
    import networkx as nx
except Exception:
    nx = None


from functools import partial

# ---------------------------
# Robust helpers
# ---------------------------

def _safe_div(a, b):
    return a / (b + 1e-8)

def _entropy_from_counts(s):
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _entropy(counts):
    tot = counts.sum()
    if tot <= 0: return 0.0
    p = counts / tot
    return float(-(p * np.log(p + 1e-12)).sum())


def _roll_window(df, ref_time, days):
    if days is None or days <= 0:
        return df
    cutoff = ref_time - pd.Timedelta(days=days)
    return df[df['dt'] >= cutoff]

def _hhi_from_counts(s):
    p = s / (s.sum() + 1e-12)
    return float((p ** 2).sum())

def safe_num(x):
    try:
        v = float(x)
        if not np.isfinite(v):
            return np.nan
        return v
    except Exception:
        return np.nan

def try_parse_time(s):
    try:
        t = datetime.strptime(str(s), "%H:%M:%S")
        return t.hour, t.minute, t.second
    except Exception:
        return np.nan, np.nan, np.nan

def try_parse_date(x):
    if isinstance(x, pd.Timestamp):
        return x
    xs = str(x)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(xs, fmt)
        except Exception:
            pass
    try:
        day = int(float(xs))
        return datetime(2025,1,1) + pd.Timedelta(days=day)
    except Exception:
        return pd.NaT

def get_best_threshold(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p * r) / (p + r + 1e-12)
    idx = np.nanargmax(f1)
    # precision_recall_curve outputs len(thr)=len(p)-1, align safely
    return float(thr[max(0, idx - 1)]) if len(thr) else 0.5


def amount_anomaly_features(df_txn: pd.DataFrame):
    df = df_txn.copy()
    df['amt'] = pd.to_numeric(df['txn_amt'], errors='coerce').fillna(0.0)

    def near_multiple(x, m, tol=10.0):
        r = np.mod(x, m)
        return (r <= tol) | (m - r <= tol)

    for m in (100, 500, 1000):
        df[f'near_{m}'] = near_multiple(df['amt'].values, float(m), tol=10.0).astype('int8')

    # Aggregate per account (outgoing)
    g = df.groupby('from_acct')
    out = pd.DataFrame({
        'acct': g.size().index.astype(str),
        'out_round100_rate': g['near_100'].mean().values,
        'out_round500_rate': g['near_500'].mean().values,
        'out_round1000_rate': g['near_1000'].mean().values,
    })

    # Daily sum to detect split structured payments near 100k buckets
    if 'txn_date' in df.columns:
        daily = df.groupby(['from_acct','txn_date'])['amt'].sum().reset_index()
        daily['near_100k'] = near_multiple(daily['amt'].values, 100000.0, tol=500.0).astype('int8')
        s = daily.groupby('from_acct')['near_100k'].mean().rename('out_split100k_day_rate').reset_index()
        out = out.merge(s.rename(columns={'from_acct':'acct'}), on='acct', how='left')
    return out.fillna(0.0)

# -------------------------
# 4) Novel partners & reciprocity dynamics
# -------------------------
def partner_novelty_features(df_txn: pd.DataFrame, windows=(7,30)):
    df = _ensure_datetime_cols(df_txn)
    df = df[['from_acct','to_acct','dt']].dropna().sort_values(['from_acct','to_acct','dt'])
    first_seen = df.groupby(['from_acct','to_acct'])['dt'].min().rename('first_seen').reset_index()
    ref = df['dt'].max()
    out = pd.DataFrame({'acct': df['from_acct'].astype(str).unique()})
    for w in windows:
        cut = ref - pd.Timedelta(days=w)
        novel = first_seen[first_seen['first_seen'] >= cut]
        denom = (df.groupby('from_acct')['to_acct'].nunique() + 1e-8)
        rate = novel.groupby('from_acct').size().reindex(denom.index).fillna(0) / denom
        out = out.merge(rate.rename(f'out_{w}d_novel_partner_rate').reset_index().rename(columns={'from_acct':'acct'}),
                        on='acct', how='left')
    out = out.fillna(0.0)

    # Reciprocity share in 30/90d
    dfR = _ensure_datetime_cols(df_txn)
    ref = dfR['dt'].max()
    for w in (30,90):
        d = _roll_window(dfR, ref, w)
        fwd_pairs = d[['from_acct','to_acct']].drop_duplicates().astype(str)
        rev_pairs = fwd_pairs.rename(columns={'from_acct':'to_acct','to_acct':'from_acct'})
        fwd = set(map(tuple, fwd_pairs[['from_acct','to_acct']].values))
        rev = set(map(tuple, rev_pairs[['from_acct','to_acct']].values))
        recip_flag = [1 if (b,a) in fwd else 0 for (a,b) in fwd]
        recip_df = pd.DataFrame(list(fwd), columns=['from_acct','to_acct'])
        recip_df['rec'] = recip_flag
        rr = recip_df.groupby('from_acct')['rec'].mean().rename(f'out_{w}d_reciprocal_share').reset_index()
        out = out.merge(rr.rename(columns={'from_acct':'acct'}), on='acct', how='left')
    return out.fillna(0.0)


from sklearn.metrics import precision_recall_curve
import numpy as np

def best_thr_min_recall(y_true, y_prob, min_recall=0.30):
    p, r, t = precision_recall_curve(y_true, y_prob)
    if len(t) == 0:
        return 0.5
    f1 = (2*p[:-1]*r[:-1])/(p[:-1]+r[:-1]+1e-12)
    ok = r[:-1] >= min_recall
    if ok.any():
        idx = np.argmax(f1 * ok)
        return float(t[idx])
    return float(t[np.argmax(f1)])


def binarize_with_topk(probs, thr, min_pos=50):
    pred = (probs >= thr).astype(int)
    if pred.sum() < min_pos:
        k = min(min_pos, len(probs))
        topk = np.argpartition(probs, -k)[-k:]
        pred[:] = 0
        pred[topk] = 1
    return pred


def _ensure_datetime_cols(df: pd.DataFrame):
    df = df.copy()
    # txn_date may be day index; attempt robust parse
    if 'txn_date' in df.columns:
        try:
            # if already int-like days, map to a reference start
            dnum = pd.to_numeric(df['txn_date'], errors='coerce')
            if dnum.notna().all():
                base = pd.Timestamp('2025-01-01')  # arbitrary epoch; relative windows only
                df['date_parsed'] = base + pd.to_timedelta(dnum, unit='D')
            else:
                df['date_parsed'] = pd.to_datetime(df['txn_date'], errors='coerce')
        except Exception:
            df['date_parsed'] = pd.to_datetime(df['txn_date'], errors='coerce')
    else:
        df['date_parsed'] = pd.NaT

    if 'txn_time' in df.columns:
        # parse HH:MM:SS robustly
        def _hms(x):
            try:
                h, m, s = str(x).split(':')
                return int(h), int(m), int(s)
            except Exception:
                return np.nan, np.nan, np.nan
        t = df['txn_time'].apply(_hms)
        df['hour'] = t.apply(lambda z: z[0]).astype('float32')
        df['minute'] = t.apply(lambda z: z[1]).astype('float32')
        df['second'] = t.apply(lambda z: z[2]).astype('float32')
    else:
        df['hour'] = np.nan; df['minute'] = np.nan; df['second'] = np.nan

    # combined timestamp
    df['dt'] = pd.to_datetime(df['date_parsed'], errors='coerce')
    df.loc[df['dt'].notna() & df['hour'].notna(), 'dt'] = df.loc[df['dt'].notna() & df['hour'].notna(), 'dt'] +         pd.to_timedelta(df.loc[df['dt'].notna() & df['hour'].notna(), 'hour'], unit='h') +         pd.to_timedelta(df.loc[df['dt'].notna() & df['hour'].notna(), 'minute'].fillna(0), unit='m') +         pd.to_timedelta(df.loc[df['dt'].notna() & df['hour'].notna(), 'second'].fillna(0), unit='s')
    return df


def conc_amount_hhi(df_txn: pd.DataFrame, windows=(7,30,90)):
    df = _ensure_datetime_cols(df_txn)
    ref = df['dt'].max()
    outs, ins = [], []
    for w in windows:
        d = _roll_window(df, ref, w)
        # outgoing
        g = d.groupby(['from_acct','to_acct'])['txn_amt'].sum()
        if len(g) > 0:
            tot = g.groupby(level=0).sum()
            share = g / tot.reindex(g.index.get_level_values(0)).values
            hhi = share.pow(2).groupby(level=0).sum().rename(f'out_{w}d_cp_hhi_amt')
            top1 = share.groupby(level=0).max().rename(f'out_{w}d_cp_top1_amt')
            outs.append(pd.concat([hhi, top1], axis=1).reset_index().rename(columns={'from_acct':'acct'}))
        # incoming
        g2 = d.groupby(['to_acct','from_acct'])['txn_amt'].sum()
        if len(g2) > 0:
            tot2 = g2.groupby(level=0).sum()
            share2 = g2 / tot2.reindex(g2.index.get_level_values(0)).values
            hhi2 = share2.pow(2).groupby(level=0).sum().rename(f'in_{w}d_cp_hhi_amt')
            top12 = share2.groupby(level=0).max().rename(f'in_{w}d_cp_top1_amt')
            ins.append(pd.concat([hhi2, top12], axis=1).reset_index().rename(columns={'to_acct':'acct'}))
    out = None
    if outs:
        out = outs[0]
        for t in outs[1:]:
            out = out.merge(t, on='acct', how='outer')
    if ins:
        tmp = ins[0]
        for t in ins[1:]:
            tmp = tmp.merge(t, on='acct', how='outer')
        out = tmp if out is None else out.merge(tmp, on='acct', how='outer')
    if out is None:
        out = pd.DataFrame({'acct':[]})
    return out.fillna(0.0)


def burstiness_features(df_txn: pd.DataFrame):
    df = _ensure_datetime_cols(df_txn)
    feats = []

    for key, prefix in [('from_acct','out'), ('to_acct','in')]:
        d = df[['dt', key]].dropna().sort_values([key, 'dt'])
        d['gap'] = d.groupby(key)['dt'].diff().dt.total_seconds()
        g = d.dropna().groupby(key)['gap']
        if len(g) == 0:
            continue
        f = pd.DataFrame({
            'acct': g.mean().index.astype(str),
            f'{prefix}_gap_mean': g.mean().values,
            f'{prefix}_gap_std': g.std().fillna(0).values,
            f'{prefix}_gap_p95': g.quantile(0.95).values,
        })
        f[f'{prefix}_gap_cv'] = _safe_div(f[f'{prefix}_gap_std'], (f[f'{prefix}_gap_mean'].abs() + 1e-6))
        f[f'{prefix}_gap_burst'] = _safe_div(f[f'{prefix}_gap_std'] - f[f'{prefix}_gap_mean'],
                                             f[f'{prefix}_gap_std'] + f[f'{prefix}_gap_mean'] + 1e-6)
        feats.append(f)

    if not feats:
        return pd.DataFrame({'acct':[]})
    out = feats[0]
    for t in feats[1:]:
        out = out.merge(t, on='acct', how='outer')
    return out.fillna(0.0)

# -------------------------
# 3) Round-amount & structuring heuristics
# -------------------------
def amount_anomaly_features(df_txn: pd.DataFrame):
    df = df_txn.copy()
    df['amt'] = pd.to_numeric(df['txn_amt'], errors='coerce').fillna(0.0)

    def near_multiple(x, m, tol=10.0):
        r = np.mod(x, m)
        return (r <= tol) | (m - r <= tol)

    for m in (100, 500, 1000):
        df[f'near_{m}'] = near_multiple(df['amt'].values, float(m), tol=10.0).astype('int8')

    # Aggregate per account (outgoing)
    g = df.groupby('from_acct')
    out = pd.DataFrame({
        'acct': g.size().index.astype(str),
        'out_round100_rate': g['near_100'].mean().values,
        'out_round500_rate': g['near_500'].mean().values,
        'out_round1000_rate': g['near_1000'].mean().values,
    })

    # Daily sum to detect split structured payments near 100k buckets
    if 'txn_date' in df.columns:
        daily = df.groupby(['from_acct','txn_date'])['amt'].sum().reset_index()
        daily['near_100k'] = near_multiple(daily['amt'].values, 100000.0, tol=500.0).astype('int8')
        s = daily.groupby('from_acct')['near_100k'].mean().rename('out_split100k_day_rate').reset_index()
        out = out.merge(s.rename(columns={'from_acct':'acct'}), on='acct', how='left')
    return out.fillna(0.0)



# -------------------------
# 9) EWMA velocity features (half-life decay)
# -------------------------
def ewma_velocity_features(df_txn: pd.DataFrame, half_lives=(1,7,30)):
    df = _ensure_datetime_cols(df_txn)
    df = df[['from_acct','to_acct','dt','txn_amt']].dropna().copy()
    df['date'] = df['dt'].dt.floor('D')
    outs = []
    # daily aggregates per sender
    daily = df.groupby(['from_acct','date'])['txn_amt'].agg(['size','sum']).reset_index()
    daily = daily.rename(columns={'size':'out_cnt','sum':'out_amt'})
    daily = daily.sort_values(['from_acct','date'])
    for hl in half_lives:
        alpha = 1 - np.exp(-np.log(2)/hl)  # convert half-life (days) to alpha
        # EWMA by group
        e = daily.groupby('from_acct')[['out_cnt','out_amt']].apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        e.columns = [f'ewma{hl}_cnt', f'ewma{hl}_amt']
        tmp = pd.concat([daily[['from_acct','date']], e.reset_index(drop=True)], axis=1)
        last = tmp.sort_values(['from_acct','date']).groupby('from_acct').tail(1)
        outs.append(last[['from_acct', f'ewma{hl}_cnt', f'ewma{hl}_amt']].rename(columns={'from_acct':'acct'}))
    out = outs[0]
    for t in outs[1:]:
        out = out.merge(t, on='acct', how='outer')
    return out.fillna(0.0)



def channel_currency_features(df_txn: pd.DataFrame, windows=(30,90)):
    df = _ensure_datetime_cols(df_txn)
    ref = df['dt'].max()
    outs = []
    for w in windows:
        d = _roll_window(df, ref, w)
        for col, pref in [('channel_type','ch'), ('currency_type','cur')]:
            if col not in d.columns: 
                continue
            # per account counts
            c = d.groupby(['from_acct', col]).size().rename('cnt')
            if len(c) == 0:
                continue
            ent = c.groupby(level=0).apply(_entropy_from_counts).rename(f'out_{w}d_{pref}_entropy')
            # HHI
            hhi = c.groupby(level=0).apply(_hhi_from_counts).rename(f'out_{w}d_{pref}_hhi')
            # KL divergence vs global (smoothed)
            glob = c.groupby(level=1).sum()
            glob = (glob + 1.0) / (glob.sum() + len(glob))   # add-1 smoothing
            def _kl(sub):
                q = glob.copy()
                p = (sub + 1.0) / (sub.sum() + len(q))
                # align
                p, q = p.align(q, join='left', fill_value=1.0/len(q))
                return float((p * np.log(p / q)).sum())
            kl = c.groupby(level=0).apply(_kl).rename(f'out_{w}d_{pref}_kl_global')
            tmp = pd.concat([ent, hhi, kl], axis=1).reset_index().rename(columns={'from_acct':'acct'})
            outs.append(tmp)
    if not outs:
        return pd.DataFrame({'acct':[]})
    out = outs[0]
    for t in outs[1:]:
        out = out.merge(t, on='acct', how='outer')
    return out.fillna(0.0)


def graph_topology_features(df_txn: pd.DataFrame, max_nodes_for_exact=200000):
    if nx is None:
        return pd.DataFrame({'acct':[]})
    # Build undirected simple graph for local structure
    df = df_txn[['from_acct','to_acct']].dropna().astype(str).copy()
    nodes = pd.Index(pd.concat([df['from_acct'], df['to_acct']]).unique())
    if len(nodes) > max_nodes_for_exact:
        # too large; return ids and skip heavy metrics
        return pd.DataFrame({'acct':nodes.values})
    G = nx.from_pandas_edgelist(df, 'from_acct', 'to_acct', create_using=nx.Graph())
    clust = nx.clustering(G)
    core = nx.core_number(G)
    try:
        btw = nx.betweenness_centrality(G, k=min(1000, max(100, int(0.01*len(G)))), seed=42)
    except Exception:
        btw = {n:0.0 for n in G.nodes()}
    out = pd.DataFrame({
        'acct': list(G.nodes()),
        'cluster_coeff': [clust.get(n,0.0) for n in G.nodes()],
        'kcore': [core.get(n,0) for n in G.nodes()],
        'betweenness_approx': [btw.get(n,0.0) for n in G.nodes()],
    })
    return out



def motif_features(edges_df):
    import networkx as nx, pandas as pd
    G = nx.from_pandas_edgelist(edges_df.astype(str), 'from_acct','to_acct', create_using=nx.DiGraph())
    # 2-cycles (A<->B)
    und = G.to_undirected()
    tri = nx.triangles(und)  # per node undirected triangles
    # reciprocal degree
    rec = {}
    for u in G.nodes():
        rec[u] = sum(1 for v in G.successors(u) if G.has_edge(v,u))
    df = pd.DataFrame({
        'acct': list(G.nodes()),
        'reciprocal_deg': [rec[u] for u in G.nodes()],
        'triangles_u': [tri.get(u,0) for u in G.nodes()],
    })
    return df


import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import cg

def ppr_to_seeds(edges_df, acct_index, seed_accts, alpha=0.85, maxiter=50, tol=1e-6):
    # Build symmetric normalized adjacency with self loops
    a2i = {a:i for i,a in enumerate(acct_index)}
    rows = edges_df['from_acct'].map(a2i).values
    cols = edges_df['to_acct'].map(a2i).values
    data = np.ones_like(rows, dtype=np.float32)
    N = len(acct_index)
    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    A = A + A.T
    A = A + sp.eye(N, dtype=np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1.0/(deg+1e-9))
    S = Dinv @ A  # random-walk matrix

    # teleport vector over seed nodes (train-fold alerts only!)
    r = np.zeros(N, dtype=np.float32)
    idx = [a2i[a] for a in seed_accts if a in a2i]
    if len(idx)==0:
        return np.zeros(N, dtype=np.float32)
    r[idx] = 1.0/len(idx)

    # solve (I - alpha S^T) x = (1-alpha) r
    M = sp.eye(N, dtype=np.float32) - alpha * S.T
    b = (1-alpha) * r
    x, _ = cg(M.tocsr(), b, maxiter=maxiter, tol=tol)
    return x  # PPR score per account


import networkx as nx
from gensim.models import Word2Vec

def node2vec_embeddings(edges_df, dim=32, walk_len=20, walks_per_node=10, window=5, min_count=1, workers=4):
    G = nx.from_pandas_edgelist(edges_df.astype(str), 'from_acct','to_acct', create_using=nx.Graph())
    nodes = list(G.nodes())
    import random
    def random_walk(start):
        walk = [start]
        for _ in range(walk_len-1):
            cur = walk[-1]
            nbrs = list(G.neighbors(cur))
            if not nbrs: break
            walk.append(random.choice(nbrs))
        return walk
    walks = []
    for _ in range(walks_per_node):
        for n in nodes:
            walks.append(random_walk(n))
    w2v = Word2Vec(sentences=walks, vector_size=dim, window=window, sg=1, hs=0, negative=5,
                   min_count=min_count, workers=workers, epochs=3)
    import numpy as np, pandas as pd
    E = np.vstack([w2v.wv[n] if n in w2v.wv else np.zeros(dim) for n in nodes])
    cols = [f"n2v_{i}" for i in range(dim)]
    return pd.DataFrame(E, columns=cols, index=nodes).reset_index().rename(columns={'index':'acct'})


import networkx as nx
from collections import Counter

def community_features(edges_df, train_alert_set):
    import networkx.algorithms.community as nxcom
    G = nx.from_pandas_edgelist(edges_df.astype(str), 'from_acct','to_acct', create_using=nx.Graph())
    parts = list(nxcom.greedy_modularity_communities(G))
    cid = {}
    for i, c in enumerate(parts):
        for n in c: cid[n] = i
    comm = pd.DataFrame({'acct': list(cid.keys()), 'comm_id': list(cid.values())})
    comm_sz = Counter(cid.values())
    comm['comm_size'] = comm['comm_id'].map(comm_sz)
    # fold-safe target-encoding: fraction of train positives in the community
    tr_pos = set(str(x) for x in train_alert_set)
    def comm_pos_rate(i):
        members = [n for n,c in cid.items() if c==i]
        return sum(1 for n in members if n in tr_pos) / max(1,len(members))
    comm['comm_pos_rate_tr'] = comm['comm_id'].map(comm_pos_rate)
    return comm



def markov_ngrams(df, key='from_acct', col='channel_type'):
    d = df[[key, 'date_parsed', col]].dropna().astype({col:str}).sort_values([key,'date_parsed'])
    d['next'] = d.groupby(key)[col].shift(-1)
    trans = d.dropna().groupby([key, col, 'next']).size().rename('cnt').reset_index()
    # normalize to conditional probs per (acct, col)
    trans['p'] = trans.groupby([key,col])['cnt'].transform(lambda s: s/s.sum())
    # take max transition prob and entropy over transitions
    import numpy as np
    agg = trans.groupby(key).agg(
        ch_trans_max=('p','max'),
        ch_trans_entropy=('p',lambda x: float(-(x*np.log(x+1e-12)).sum()))
    ).reset_index().rename(columns={key:'acct'})
    return agg.fillna(0.0)


# ---------------------------
# Feature engineering
# ---------------------------
def build_features_from_transactions(txn,ref_date=None):
    """
    Returns an account-level dataframe `df` with:
      - acct (string id)
      - is_esun (1/0 if present in source; else defaults to 1)
      - a set of simple but strong aggregates (in/out counts, sums, uniq partners, time features)
    """
    df = txn.copy()
   

    import networkx as nx
    G = nx.from_pandas_edgelist(df, 'from_acct', 'to_acct', 
                            edge_attr='txn_amt', create_using=nx.DiGraph())
    pr = nx.pagerank(G, weight='txn_amt')
    
    if "txn_time" in df.columns:
        hhmmss = df["txn_time"].apply(try_parse_time)
        t = pd.to_datetime(df['txn_time'], format='%H:%M:%S', errors='coerce') 
        df["hour"] = hhmmss.apply(lambda t: t[0])
        df['minute'] = t.dt.minute.fillna(-1).astype('int16')
        df['second'] = t.dt.second.fillna(-1).astype('int16')
        df['is_night']     = ((df['hour'] >= 0) & (df['hour'] < 6)).astype('int8')
        df['is_morning']   = ((df['hour'] >= 6) & (df['hour'] < 12)).astype('int8')
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype('int8')
        df['is_evening']   = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype('int8')
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
        df['is_lunch_window']   = ((df['hour'] == 12) | (df['hour'] == 13)).astype(int)
        df['sec_since_midnight'] = (df['hour'].clip(lower=0) * 3600 + df['minute'].clip(lower=0) * 60 + df['second'].clip(lower=0)).where(df['hour'] >= 0, np.nan)
        df['fraction_of_day']    = (df['sec_since_midnight'] / 86400).fillna(-1.0)
        grp = df.groupby('txn_date' , dropna=False)['txn_amt']
        mean_by_day = grp.transform('mean')
        std_by_day  = grp.transform('std').fillna(0)
        df['amt_z_by_day'] = ((df['txn_amt'] - mean_by_day) / (std_by_day + 1e-6)).replace([np.inf, -np.inf], 0).fillna(0)

     

    if "is_self_txn" in df.columns:
        df["is_self_txn"] = df["is_self_txn"].map({"Y":1, "N":0}).fillna(0).astype(float)
    else:
        df["is_self_txn"] = 0.0

    cat_cols = ["currency_type","channel_type"]

    for cols in cat_cols:
        le.fit(df[cols])
        data_cat=le.transform(df[cols])
        df[cols] = data_cat      

    if "txn_date" in df.columns:
        dt = df["txn_date"].apply(try_parse_date)
        df["date_parsed"] = pd.to_datetime(dt, errors="coerce")
        df["dow"] = df["date_parsed"].dt.dayofweek
        df["is_weekend"] = df["dow"].isin([5,6]).astype(float)
    else:
        df["dow"] = np.nan
        df["is_weekend"] = np.nan

    
    twohop = _two_hop_features(df)
    
    g_from = df.groupby("from_acct")
    out = pd.DataFrame({
        "acct": g_from.size().index,
        "out_txn_cnt": g_from.size().values,
        "out_amt_sum": g_from["txn_amt"].sum().values,
        "out_amt_mean": g_from["txn_amt"].mean().values,
        "out_amt_max": g_from["txn_amt"].max().values, 
        "out_amt_median": g_from["txn_amt"].median().values,
        "out_amt_std": g_from["txn_amt"].std().fillna(0).values,
        "out_hour_mean": g_from["hour"].mean().values,
        "out_minute_mean": g_from["minute"].mean().values,
        "out_morning_mean": g_from["is_morning"].mean().values,
        "out_night_mean": g_from["is_night"].mean().values,
        "out_is_lunch_window_mean": g_from["is_lunch_window"].mean().values,
        "out_is_is_business_hours_mean": g_from["is_business_hours"].mean().values,
        "out_eve_mean": g_from["is_evening"].mean().values,
        "out_biznis_mean": g_from["is_business_hours"].mean().values,
        "out_lunch_mean": g_from["is_lunch_window"].mean().values,
        "out_self_rate": g_from["is_self_txn"].mean().values,
        "out_ch": g_from["channel_type"].mean().values,
        "out_curr": g_from["currency_type"].mean().values,
        "out_date": g_from["txn_date"].mean().values,
        "out_dow": g_from["dow"].mean().values,
        "out_weekend": g_from["is_weekend"].mean().values,
        "out_fraction_day": g_from["fraction_of_day"].mean().values, 
        "out_amt_day": g_from["amt_z_by_day"].mean().values  
        })
 
    # -------- Inbound aggregates (by to_acct)
    g_to = df.groupby("to_acct")
    inn = pd.DataFrame({
        "acct": g_to.size().index,
        "in_txn_cnt": g_to.size().values,
        "in_amt_sum": g_to["txn_amt"].sum().values,
        "in_amt_mean": g_to["txn_amt"].mean().values,
        "in_amt_std": g_to["txn_amt"].std().fillna(0).values,
        "in_amt_max": g_to["txn_amt"].max().fillna(0).values,
        "in_hour_mean": g_to["hour"].mean().values,
        "in_minute_mean": g_to["minute"].mean().values,
        "in_morning_mean": g_to["is_morning"].mean().values,
        "in_night_mean": g_to["is_night"].mean().values,
        "in_is_lunch_window_mean": g_to["is_lunch_window"].mean().values,
        "in_is_is_business_hours_mean": g_to["is_business_hours"].mean().values,
        "in_eve_mean": g_to["is_evening"].mean().values,
        "in_biznis_mean": g_to["is_business_hours"].mean().values,
        "in_lunch_mean": g_to["is_lunch_window"].mean().values,
        "in_self_rate": g_to["is_self_txn"].mean().values,
        "in_ch": g_to["channel_type"].mean().values,
        "in_curr": g_to["currency_type"].mean().values,
        "in_date": g_to["txn_date"].mean().values,
        "in_dow": g_to["dow"].mean().values,
        "in_weekend": g_to["is_weekend"].mean().values,
        "in_fraction_day": g_to["fraction_of_day"].mean().values, 
        "in_amt_day": g_to["amt_z_by_day"].mean().values  
        })

    
    df_from = df[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'})
    df_to = df[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'})
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates().reset_index(drop=True)
    

    df_acct = pd.merge(out, inn, on="acct", how="outer").fillna(0) 
    df_acct['pagerank'] = df_acct['acct'].map(pr).fillna(0)

    df_acct["io_txn_ratio"] = df_acct["in_txn_cnt"] / (df_acct["out_txn_cnt"] + 1e-6)
    df_acct["io_amt_ratio"] = df_acct["in_amt_sum"] / (df_acct["out_amt_sum"] + 1e-6)
    df_acct["total_txn"] = df_acct["in_txn_cnt"] + df_acct["out_txn_cnt"]
    df_acct["total_amt"] = df_acct["in_amt_sum"] + df_acct["out_amt_sum"]

    df_acct['in_gaussian'] = df_acct.groupby('acct')['in_amt_mean'].transform(partial(gaussian_from_onehot, sigma=15, m=100)  ) 

    df_acct['out_gaussian'] = df_acct.groupby('acct')['out_amt_mean'].transform(partial(gaussian_from_onehot, sigma=15, m=100)  ) 


    columns = df_acct.drop("acct", axis=1).columns.tolist()
         
    pca = PCA(random_state=42,n_components=1)

    pg_features =  df_acct.filter(regex='out.*')
    train_pca = pca.fit_transform(pg_features)
    df_acct['PCA_OUT'] = train_pca[:,0]
    
    in_features =  df_acct.filter(regex='in.*')
    train_pca_in = pca.fit_transform(pg_features)
    df_acct['PCA_IN'] = train_pca[:,0]


    send_stats = _agg_by(df.rename(columns={'from_acct': 'acct'}), 'acct', 'txn_amt', 'send')
    recv_stats = _agg_by(df.rename(columns={'to_acct':   'acct'}), 'acct', 'txn_amt', 'recv')

     
    for col in columns:
        ranks = rank_4_3_2_1(df_acct[col])
        df_acct[f"{col}_rank"] = ranks
   
    df_acct = df_acct.merge(twohop, on='acct', how='left') 
    df_acct = df_acct.merge(send_stats, on='acct', how='left') 
    df_acct = df_acct.merge(recv_stats, on='acct', how='left') 
   
    df_acct = df_acct.merge(burstiness_features(df),on='acct',how='left')
    df_acct = df_acct.merge(amount_anomaly_features(df),on='acct',how='left')
    df_acct = df_acct.merge(conc_amount_hhi(df, windows=(7,30,90)),on='acct',how='left')
    df_acct = df_acct.merge(conc_amount_hhi(df, windows=(7,30,90)),on='acct',how='left')
    df_acct = df_acct.merge(channel_currency_features(df, windows=(7,30,90)),on='acct',how='left')
    df_acct = df_acct.merge(graph_topology_features(df),on="acct",how="left")
    

    
    df_acct = df_acct.merge(partner_novelty_features(df,windows=(7,30)),on="acct",how="left")
    df_acct = df_acct.merge(channel_currency_features(df,windows=(30,90)),on="acct",how="left")
    df_acct = df_acct.merge(ewma_velocity_features(df, half_lives=(1,7,30)),on="acct",how="left")
    df_acct = df_acct.merge(motif_features(df),on="acct",how="left")

 
    edges = df[['from_acct','to_acct']].dropna().astype(str)
    acct_index = pd.Index(pd.concat([edges['from_acct'], edges['to_acct']]).unique())

    # PPR (per fold, seeds = train positives in that fold)
    #ppr = ppr_to_seeds(edges, acct_index, seed_accts=train_pos_set)
    #df_acct = df_acct.merge(pd.DataFrame({'acct':acct_index, 'ppr_to_pos':ppr}), on='acct', how='left')

    
    n2v = node2vec_embeddings(df[['from_acct','to_acct']], dim=32)
    df_acct = df_acct.merge(n2v, on='acct', how='left')

    # Motifs + Community
    df_acct = df_acct.merge(motif_features(df), on='acct', how='left')
        # Channel sequence n-grams
    df_acct = df_acct.merge(markov_ngrams(df, key='from_acct', col='channel_type'), on='acct', how='left')


    df_result = pd.merge(df_acct, df_acc, on='acct', how='left')
    

    return df_result


def _agg_by(df: pd.DataFrame, key: str, val: str, prefix: str) -> pd.DataFrame:
    g = df.groupby(key)[val]
    stats = {
        f'{prefix}_sum': g.sum(),
        f'{prefix}_mean': g.mean(),
        f'{prefix}_std': g.std(ddof=0),
        f'{prefix}_min': g.min(),
        f'{prefix}_max': g.max(),
        f'{prefix}_cnt': g.count(),
    }
    q = g.quantile([0.25, 0.5, 0.75]).unstack()
    stats[f'{prefix}_q25'] = q[0.25]
    stats[f'{prefix}_q50'] = q[0.5]
    stats[f'{prefix}_q75'] = q[0.75]
    out = pd.DataFrame(stats).fillna(0.0).reset_index()
    return out.rename(columns={key: 'acct'})



def rank_4_3_2_1(x):
    # Remove outliers using 3 standard deviations rule
    no_outliers = x[(x - x.mean()).abs() <= 3 * x.std()]
    q1 = no_outliers.quantile(0.25)
    q2 = no_outliers.quantile(0.5)
    q3 = no_outliers.quantile(0.75)
    ranks = pd.Series(index=x.index)
    ranks[x >= q3] = 2
    ranks[(x >= q2) & (x < q3)] = 3
    ranks[(x >= q1) & (x < q2)] = 4
    ranks[x < q1] = 1
    return ranks

def _neighbor_degree(df: pd.DataFrame) -> pd.DataFrame:
    out_deg = df.groupby('from_acct')['to_acct'].nunique()
    out_deg = out_deg.rename('out_degree')
    to_out = df[['from_acct','to_acct']].merge(out_deg.rename_axis('acct').reset_index(), left_on='to_acct', right_on='acct', how='left')
    to_out['out_degree'] = to_out['out_degree'].fillna(0)
    neigh_avg = to_out.groupby('from_acct')['out_degree'].mean().reset_index().rename(columns={'from_acct':'acct','out_degree':'neighbor_outdeg_mean'})
    return neigh_avg


def _two_hop_features(df: pd.DataFrame, max_neighbors:int=200) -> pd.DataFrame:
    g = df.groupby('from_acct')['to_acct'].apply(lambda s: set(s.values)).to_dict()
    two_hop_counts = {}
    for a, n1 in g.items():
        if len(n1) == 0:
            two_hop_counts[a] = 0
            continue
        n1_sample = list(n1)[:max_neighbors]
        n2 = set()
        for b in n1_sample:
            n2 |= g.get(b, set())
            if len(n2) > 5000:
                break
        two_hop_counts[a] = max(0, len(n2 - n1 - {a}))
    return pd.DataFrame({'acct': list(two_hop_counts.keys()), 'two_hop_unique': list(two_hop_counts.values())})



# ---------------------------
# Your Train/Test split
# ---------------------------
def TrainTestSplit(df, df_alert, df_test):
    """
    y_train = 1 if acct appears in df_alert['acct'], else 0.
    Train = (~in test) & (is_esun==1)
    Test  = in test list (drop 'is_esun' from features)
    """
    # Normalize ids to str for consistent join
    df = df.copy()
    df["acct"] = df["acct"].astype(str)
    df_alert = df_alert.copy()
    df_alert["acct"] = df_alert["acct"].astype(str)
    df_test = df_test.copy()
    df_test["acct"] = df_test["acct"].astype(str)

    X_train = df[(~df["acct"].isin(df_test["acct"])) & (df["is_esun"] == 1)].drop(columns=["is_esun"]).copy()
    y_train = X_train["acct"].isin(df_alert["acct"]).astype(int)
    X_test  = df[df["acct"].isin(df_test["acct"])].drop(columns=["is_esun"]).copy()

    # Keep ids for submission, drop acct from features
    train_ids = X_train["acct"].values
    test_ids  = X_test["acct"].values
    X_train = X_train.drop(columns=["acct"])
    X_test  = X_test.drop(columns=["acct"])

    pos = int(y_train.sum())
    neg = int((1 - y_train).sum())
    print(f"(Finish) Train-Test-Split | train={len(y_train)} (pos={pos}, neg={neg}, pos_rate={pos/(pos+neg+1e-9):.4f}) | test={len(X_test)}")
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True).astype(int).to_numpy()
    return X_train, X_test, y_train, train_ids, test_ids


def gaussian_density(n, sigma):
    x = np.arange(n)
    mu = n // 2
    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_from_onehot(x, sigma, m=1):
    """
    Calculates the probability density of a Gaussian distribution based on a one-hot encoded array.

    Args:
        x: A 1D numpy array representing a one-hot encoded vector.  Only one element
           should be 1, and the rest should be 0.  The position of the '1'
           indicates the mean of the Gaussian.
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        A 1D numpy array of the same shape as x, containing the probability density
        values of the Gaussian distribution.  Values beyond 3*sigma from the mean
        are set to 0.
    """

    if not isinstance(x, np.ndarray):
        x = x.values
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    if np.sum(x) != 1:
        return x*0
    if not np.all((x == 0) | (x == 1)):
        raise ValueError("x must contain only 0s and 1s")
    if not isinstance(sigma, (int, float)):
         raise TypeError("sigma must be a number")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Find the index of the '1' (the mean)
    mean_index = np.where(x == 1)[0][0]

    # Create an array of indices corresponding to the positions in x
    indices = np.arange(len(x))

    # Calculate the Gaussian probability density
    y = norm.pdf(indices, loc=mean_index, scale=sigma)

    # Set values beyond 3*sigma to 0
    distance_from_mean = np.abs(indices - mean_index)
    y[distance_from_mean > 3 * sigma] = 0

    return y*m

# ---------------------------
# Training loop (CV + F1 thr)
# ---------------------------
def train_and_predict(X, y, X_test, folds=5, seed=42):
    
       
    model = XGBClassifier(
            n_estimators=1500,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            colsample_bylevel=0.7,
            colsample_bynode=0.7,
            reg_lambda=2.0,
            reg_alpha=0.5,
            min_child_weight=3,
            gamma=0.3,
            max_delta_step=1,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1
        )

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    thrs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

             
        n_pos = max(int((y_tr == 1).sum()), 1)
        n_neg = int((y_tr == 0).sum())
        spw = max(n_neg / n_pos, 1.0)
        model.set_params(scale_pos_weight=spw)
            
        model.fit(X_tr, y_tr)
    
        va_prob = model.predict(X_va)
        f1 = f1_score(y_va, va_prob)
        print(f"[Fold {fold}] F1={f1:.4f}")

        oof[va_idx] = va_prob
        test_pred += model.predict(X_test) / skf.n_splits

    
    oof_f1 = f1_score(y, oof)
    print(f"[OOF] F1={oof_f1:.4f}")
    return test_pred

# ---------------------------
# Main
# ---------------------------
def main(args):
    data_dir = args.data_dir

    df_txn = pd.read_csv(os.path.join(data_dir, "acct_transaction.csv"))
    df_alert = pd.read_csv(os.path.join(data_dir, "acct_alert.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "acct_predict.csv"))

    # 1) Build account-level features
    df = build_features_from_transactions(df_txn)

    # 2) Split & labels (uses your function)
    X_train, X_test, y_train, train_ids, test_ids = TrainTestSplit(df, df_alert, df_test)

    # 3) Train & predict with F1-optimized threshold
    #test_prob, thr = train_and_predict(X_train, y_train, X_test, folds=args.folds, seed=args.seed)
    
    test_prob = train_and_predict(X_train, y_train, X_test, folds=args.folds, seed=args.seed)


    out_path = os.path.join(data_dir, "submission.csv")
    print(test_prob)
    
    #labels = binarize_with_topk(test_prob, thr, min_pos=500)  # tune 20/50/100
    labels = (test_prob >= 0.5).astype(int)

    sub = pd.DataFrame({"acct": test_ids, "label": labels.astype(int)})
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print("Test positives:", int(labels.sum()), "/", len(labels))

    '''
    # 4) Create submission
    sub = pd.DataFrame({
        "acct": test_ids,
        "label": (test_prob >= thr).astype(int)
    })
    
    sub.to_csv(out_path, index=False, encoding="utf-8")
    '''
    print(f"Saved submission → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)

