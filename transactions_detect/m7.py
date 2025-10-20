"""
2025玉山人工智慧挑戰賽 — 強化版基線程式
- 更完整的帳戶特徵工程（收/支統計、唯一往來數、淨額、比例、互惠等）
- 嚴謹的 Train/Valid 交叉驗證（StratifiedKFold, ROC-AUC/PR-AUC）
- 類別失衡處理（class_weight 或 sample_weight）
- 隨機搜尋 RandomForest 超參數
- 產出兩個檔：
    1) result.csv（TBrain 上傳用：acct,label）
    2) result_with_prob.csv（含 acct, prob, label，方便檢視）
依賽方 baseline 欄位，不假設多餘欄位（例如交易時間），若不存在會自動跳過對應特徵。
"""

import os
import warnings
import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb

RANDOM_STATE = 42
N_JOBS = -1
warnings.filterwarnings("ignore")


# =========================
# 1) Data Loading
# =========================
def LoadCSV(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    讀取挑戰賽提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    備註：不做任何列過濾；dtype 與 low_memory 做穩健載入
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'), low_memory=False)
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'), low_memory=False)
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'), low_memory=False)

    # 基本欄位檢查
    req_cols = {'from_acct', 'to_acct', 'txn_amt', 'from_acct_type', 'to_acct_type'}
    missing = req_cols - set(df_txn.columns)
    if missing:
        raise ValueError(f"交易資料缺少必要欄位: {missing}")

    # 類型處理
    for c in ['from_acct', 'to_acct']:
        if df_txn[c].dtype != 'object':
            df_txn[c] = df_txn[c].astype(str)

    if 'acct' in df_alert.columns and df_alert['acct'].dtype != 'object':
        df_alert['acct'] = df_alert['acct'].astype(str)
    if 'acct' in df_test.columns and df_test['acct'].dtype != 'object':
        df_test['acct'] = df_test['acct'].astype(str)

    # 金額容錯
    df_txn['txn_amt'] = pd.to_numeric(df_txn['txn_amt'], errors='coerce').fillna(0.0)

    print("(Finish) Load Dataset.")
    return df_txn, df_alert, df_test


# =========================
# 2) Feature Engineering
# =========================
def _agg_by(df: pd.DataFrame, key: str, val: str, prefix: str) -> pd.DataFrame:
    """
    通用彙總：sum / mean / std / min / max / count
    """
    g = df.groupby(key)[val]
    
    # 基本統計
    stats = {
        f'{prefix}_sum': g.sum(),
        f'{prefix}_mean': g.mean(),
        f'{prefix}_std': g.std(),
        f'{prefix}_min': g.min(),
        f'{prefix}_max': g.max(),
        f'{prefix}_cnt': g.count(),
    }
    
    # 分位數
    quantiles = g.quantile([0.25, 0.5, 0.75]).unstack()
    stats[f'{prefix}_q25'] = quantiles[0.25]
    stats[f'{prefix}_q50'] = quantiles[0.5]
    stats[f'{prefix}_q75'] = quantiles[0.75]

    out = pd.DataFrame(stats).fillna(0.0).reset_index().rename(columns={key: 'acct'})
    return out


def safe_div(numer, denom, default=0.0):
    """ 安全除法，避免 0 除錯誤 """
    return np.divide(numer, denom, out=np.full_like(numer, default, dtype=float), where=(denom!=0))

def clean_numeric_df(df: pd.DataFrame, clip_quantile: float = 0.999) -> pd.DataFrame:
    """取代 infs, 裁剪極端值, 確保 float64."""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        hi = out[num_cols].quantile(clip_quantile).astype(float)
        lo = out[num_cols].quantile(1.0 - clip_quantile).astype(float)
        out[num_cols] = out[num_cols].clip(lower=lo, upper=hi, axis=1)
        out[num_cols] = out[num_cols].astype('float64')
    return out


import numpy as np
import pandas as pd

def add_time_features(df, col='txn_time'):
    # parse time; coerce invalid → NaT
    dt = pd.to_datetime(df[col], format='%H:%M:%S', errors='coerce')

    # base
    df['hour']   = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['second'] = dt.dt.second

    # derived
    seconds_in_day = 24 * 60 * 60
    df['seconds_since_midnight'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
    df['minute_of_day']          = df['hour'] * 60 + df['minute']

    # buckets
    df['five_min_bucket']   = (df['minute'] // 5).astype('Int64')      # 0..11
    df['quarter_hour_bin']  = (df['minute'] // 15).astype('Int64')     # 0..3
    df['hour_bin_label']    = df['hour'].map(lambda h: f'{h:02d}:00-{(h+1)%24:02d}:00')

    # flags
    df['is_morning']        = ((df['hour'] >= 5) & (df['hour'] < 12)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
    df['is_lunch_window']   = ((df['hour'] == 12) | ((df['hour'] == 13) & (df['minute'] < 60))).astype(int)

    # cyclical encodings
    angle = 2 * np.pi * df['seconds_since_midnight'] / seconds_in_day
    df['sin_time'] = np.sin(angle)
    df['cos_time'] = np.cos(angle)

    return df

def _mean_abs_dev(s):
    s = pd.Series(s, dtype='float64')
    m = s.mean()
    return (s - m).abs().mean()


def _dir_agg(df, acct_col, prefix):
    g = df.groupby(acct_col)

    # Amount shape
    med   = g['txn_amt'].median().rename(f'{prefix}_median')
    #mad   = (g['txn_amt'].max()).rename(f'{prefix}_mad')               # mean abs dev
    mad = g['txn_amt'].apply(_mean_abs_dev).rename(f'{prefix}_mad')
    iqr   = (g['txn_amt'].quantile(0.75) - g['txn_amt'].quantile(0.25)).rename(f'{prefix}_iqr')
    zeros = (g['txn_amt'].apply(lambda s: (s==0).mean())).rename(f'{prefix}_zero_frac')

    # Time-of-day behavior (built earlier on rows)
    biz   = g['is_business_hours'].mean().rename(f'{prefix}_biz_frac')
    lunch = g['is_lunch_window'].mean().rename(f'{prefix}_lunch_frac')
    morn  = g['is_morning'].mean().rename(f'{prefix}_morn_frac')

    # Cyclical smoothness (variance of sin/cos across rows)
    sinv  = g['sin_time'].var().rename(f'{prefix}_sin_var')
    cosv  = g['cos_time'].var().rename(f'{prefix}_cos_var')

    # Unique counterpart & diversity
    if prefix == 'send':
        uniq = g['to_acct'].nunique().rename(f'{prefix}_uniq_cntp')
        ent  = g['to_acct'].apply(lambda s: (s.value_counts(normalize=True)
                                             .pipe(lambda p: -(p*np.log(p+1e-12))).sum())
                                 ).rename(f'{prefix}_cntp_entropy')
    else:
        uniq = g['from_acct'].nunique().rename(f'{prefix}_uniq_cntp')
        ent  = g['from_acct'].apply(lambda s: (s.value_counts(normalize=True)
                                               .pipe(lambda p: -(p*np.log(p+1e-12))).sum())
                                   ).rename(f'{prefix}_cntp_entropy')

    # Amount banding (micro / mid / large)
    def bands(x):
        x = pd.Series(x)
        return pd.Series({
            f'{prefix}_micro_frac': (x <= 1000).mean(),
            f'{prefix}_mid_frac': ((x > 1000) & (x <= 10000)).mean(),
            f'{prefix}_large_frac': (x > 10000).mean(),
        })
    band = g['txn_amt'].apply(bands)

    out = pd.concat([med, mad, iqr, zeros, biz, lunch, morn, sinv, cosv, uniq, ent, band], axis=1).reset_index()
    out = out.rename(columns={acct_col: 'acct'})
    return out


def PreProcessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    帳戶層級特徵：
    - 轉出/轉入 金額彙總（sum/mean/std/min/max/count）
    - 唯一往來對手數（out_degree/in_degree）
    - 淨額、比例、方向性
    - 互惠（是否同時有收有付）
    - is_esun（由 from/to 的 acct_type 推估，只要出現過 1 視為玉山戶）
    若有時間欄（例如 txn_time）可在此補上時間窗聚合；為保通用性，此版不假設存在。
    """
    # 轉出/轉入金額統計
    
    #time_feats=add_time_features(df).reset_index().rename(columns={'from_acct': 'acct'}) 
    

    dt = pd.to_datetime(df['txn_time'], format='%H:%M:%S', errors='coerce')

    # base
    df['hour']   = dt.dt.hour
    df['minute'] = dt.dt.minute
    df['second'] = dt.dt.second

    # derived
    seconds_in_day = 24 * 60 * 60
    df['seconds_since_midnight'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
    df['minute_of_day']          = df['hour'] * 60 + df['minute']

    
    

        # buckets
    df['five_min_bucket']   = (df['minute'] // 5).astype('Int64')      # 0..11
    df['quarter_hour_bin']  = (df['minute'] // 15).astype('Int64')     # 0..3
    df['hour_bin_label']    = df['hour'].map(lambda h: f'{h:02d}:00-{(h+1)%24:02d}:00')

    # flags
    df['is_morning']        = ((df['hour'] >= 5) & (df['hour'] < 12)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17)).astype(int)
    df['is_lunch_window']   = ((df['hour'] == 12) | ((df['hour'] == 13) & (df['minute'] < 60))).astype(int)

    # cyclical encodings
    angle = 2 * np.pi * df['seconds_since_midnight'] / seconds_in_day
    df['sin_time'] = np.sin(angle)
    df['cos_time'] = np.cos(angle)

    print(df['txn_date']) 

    send_stats = _agg_by(df.rename(columns={'from_acct': 'acct'}), 'acct', 'txn_amt', 'send')
    recv_stats = _agg_by(df.rename(columns={'to_acct': 'acct'}),   'acct', 'txn_amt', 'recv')

    # 唯一往來對手數
    out_deg = (
        df.groupby('from_acct')['to_acct'].nunique()
        .rename('out_unique_cnt').reset_index().rename(columns={'from_acct': 'acct'})
    )
    in_deg = (
        df.groupby('to_acct')['from_acct'].nunique()
        .rename('in_unique_cnt').reset_index().rename(columns={'to_acct': 'acct'})
    )

    # 是否同時有收有付（互惠）
    has_send = df['from_acct'].value_counts().rename('has_send').reset_index()
    has_send.columns = ['acct', 'has_send']
    has_send['has_send'] = 1

    has_recv = df['to_acct'].value_counts().rename('has_recv').reset_index()
    has_recv.columns = ['acct', 'has_recv']
    has_recv['has_recv'] = 1

    # is_esun：只要任一方向出現過 type==1 即視為玉山戶；其餘補 0
    df_from = df[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'from_is_esun'})
    df_to   = df[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'to_is_esun'})
    esun = (
        pd.concat([df_from, df_to], ignore_index=True)
        .groupby('acct')[['from_is_esun', 'to_is_esun']].max().fillna(0).reset_index()
    )
    esun['is_esun'] = ((esun['from_is_esun'] == 1) | (esun['to_is_esun'] == 1)).astype(int)
    esun = esun[['acct', 'is_esun']]

    # 合併
    feats = (
        send_stats.merge(recv_stats, on='acct', how='outer')
        .merge(out_deg, on='acct', how='outer')
        .merge(in_deg,  on='acct', how='outer')
        .merge(has_send, on='acct', how='left')
        .merge(has_recv, on='acct', how='left')
        .merge(esun, on='acct', how='left')
        .fillna(0)
    )

    # 派生特徵
    feats['total_txn_cnt'] = feats['send_cnt'] + feats['recv_cnt']
    feats['net_amt'] = feats['send_sum'] - feats['recv_sum']
    feats['gross_amt'] = feats['send_sum'] + feats['recv_sum']
    
    # 派生特徵（安全版本）
    feats['send_recv_ratio'] = safe_div(feats['send_sum'], feats['recv_sum'])
    feats['degree_ratio'] = safe_div(feats['out_unique_cnt'], feats['in_unique_cnt'])
    feats['reciprocal_flag'] = ((feats['has_send'] > 0) & (feats['has_recv'] > 0)).astype(int)
    feats['net_over_gross'] = safe_div(feats['net_amt'], feats['gross_amt'])

    send_more = _dir_agg(df, 'from_acct', 'send')
    recv_more = _dir_agg(df, 'to_acct',   'recv')

    feats = feats.merge(send_more, on='acct', how='left') \
             .merge(recv_more, on='acct', how='left') \
             .fillna(0.0)

    # Derived stability/ratio signals
    feats['amt_iqr_ratio'] = safe_div(feats['send_iqr'] + feats['recv_iqr'], feats['send_median'] + feats['recv_median'])
    feats['time_consistency'] = 1.0 / (1e-6 + feats['send_sin_var'] + feats['recv_sin_var'] + feats['send_cos_var'] + feats['recv_cos_var'])

# Log-safe transforms for new positives
    for col in ['send_median','recv_median','send_iqr','recv_iqr','send_uniq_cntp','recv_uniq_cntp']:
        feats[f'log1p_{col}'] = np.log1p(feats[col].astype('float64'))

    feats = clean_numeric_df(feats)

    # Log 轉換（穩定化）
    for col in ['send_sum', 'recv_sum', 'gross_amt', 'out_unique_cnt', 'in_unique_cnt', 'total_txn_cnt']:
        feats[f'log1p_{col}'] = np.log1p(feats[col].astype('float64'))

    # 清理數值
    feats = clean_numeric_df(feats)
    
    print_cols = [x for x in feats.columns if x not in  ['acct']]
    print(print_cols)

    #features_cat= [col for col in X_train.columns if col not in not_cols+['Month','Year','year_month']]

 
    print(feats[print_cols])
    print(print_cols)    
    print("(Finish) PreProcessing.")
    return feats


# =========================
# 3) Split & Label
# =========================
def TrainTestSplit(df_feat: pd.DataFrame, df_alert: pd.DataFrame, df_test: pd.DataFrame):
    """
    只用玉山帳戶訓練（與官方 baseline 一致），測試集為提供之 acct_predict 清單
    y_train：acct 是否在警示名單
    """
    df_feat = df_feat.copy()
    #df_feat= df_feat.fillna(0)
    
    df_alert = df_alert.copy()
    df_test = df_test.copy()
    
    #df_alert= df_alert.fillna(0)
    #df_test = df_test.fillna(0)
    


    # 訓練集：非測試清單 & is_esun==1
    X_train = df_feat[(~df_feat['acct'].isin(df_test['acct'])) & (df_feat['is_esun'] == 1)].copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)

    # 測試集：predict 清單
    X_test = df_feat[df_feat['acct'].isin(df_test['acct'])].copy()

    print(len(X_test))
    print(len(X_train))
    
    # 警示比例參考
    pos_rate = y_train.mean() if len(y_train) else 0.0
    print(f"(Finish) Train-Test-Split | train={len(X_train)}, test={len(X_test)}, positive_rate={pos_rate:.4f}")
    return X_train, X_test, y_train


# =========================
# 4) Modeling with CV & HPO
# =========================
def _make_features_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    丟掉ID與明顯不當作特徵的欄位，回傳 X 與使用到的欄位名稱
    """
    drop_cols = ['acct']
    if 'is_esun' in df.columns:  # 訓練時可保留，也可丟掉；這裡保留（對測試集也可用）
        # 留著不丟；若不想用可把它加到 drop_cols
        pass
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return df[feature_cols], feature_cols


def Modeling(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, random_state: int = RANDOM_STATE):
    """
    LightGBM + RandomizedSearchCV
    - 以 StratifiedKFold=5 做 AUC 與 PR-AUC 監控
    - 使用 is_unbalance=True 處理類別失衡
    - OOF 預測用於尋找最佳門檻
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    Xtr, feat_cols = _make_features_matrix(X_train)
    Xte, _ = _make_features_matrix(X_test)

    # 簡單補值
    imputer = SimpleImputer(strategy='median')
    Xtr_ = imputer.fit_transform(Xtr)
    Xte_ = imputer.transform(Xte)

    # 模型與搜尋空間

    # 以最佳參數再做 5-fold 檢視 PR-AUC 並收集 OOF 預測
    oof_pred = np.zeros(len(Xtr_))
    for fold, (tr_idx, va_idx) in enumerate(cv.split(Xtr_, y_train), 1):
        mdl = lgb.LGBMClassifier()
        mdl.fit(Xtr_[tr_idx], y_train.iloc[tr_idx])
        oof_pred[va_idx] = mdl.predict_proba(Xtr_[va_idx])[:, 1]
        roc = roc_auc_score(y_train.iloc[va_idx], oof_pred[va_idx])
        pr = average_precision_score(y_train.iloc[va_idx], oof_pred[va_idx])
        print(f"(CV Fold {fold}) ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

    oof_roc = roc_auc_score(y_train, oof_pred)
    oof_pr = average_precision_score(y_train, oof_pred)
    print(f"(OOF) ROC-AUC={oof_roc:.4f}, PR-AUC={oof_pr:.4f}")

    # 尋找最佳門檻
    thresholds = np.linspace(0.01, 0.99, 100)
    f1_scores = [f1_score(y_train, (oof_pred >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold from OOF: {best_threshold:.4f} (F1-score: {np.max(f1_scores):.4f})")
    print(stop)
    
    # 以全部訓練資料重新訓練
    best_model.fit(Xtr_, y_train)

    # 測試集預測機率
    y_prob = best_model.predict_proba(Xte_)[:, 1]
    # 使用最佳門檻
    y_pred = (y_prob >= best_threshold).astype(int)

    return y_pred, y_prob, feat_cols, best_model, imputer


# =========================
# 5) Output
# =========================
def OutputCSV(path: str, df_test: pd.DataFrame, X_test: pd.DataFrame, y_pred: np.ndarray):
    """
    產出兩份檔案：
      - result.csv（TBrain 上傳用）
      - result_with_prob.csv（含預測機率，方便檢視與後續調整門檻）
    """
    df_pred = pd.DataFrame({
        'acct': X_test['acct'].values,
        'label': y_pred
    })
    df_out = df_test[['acct']].merge(df_pred, on='acct', how='left')
    df_out.to_csv(path, index=False)
    print(f"(Finish) Output saved to {path}")

    
# =========================
# 6) Main
# =========================
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold,StratifiedKFold ,GroupKFold
        # 依你的資料所在目錄調整
    dir_path = "data/"
    all_tp=[]

    df_txn, df_alert, df_test = LoadCSV(dir_path)
    df_feat = PreProcessing(df_txn)
        

    X_train, X_test, y_train = TrainTestSplit(df_feat, df_alert, df_test)

    print(X_train.head(10))
    
    from catboost import CatBoostClassifier
    import re

    #training
    #cat_model =  CatBoostClassifier(**{'verbose':0}) 
    
    pos_w = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
    clf = CatBoostClassifier(
    depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
    loss_function='Logloss', eval_metric='AUC', verbose=0,
    class_weights=[1.0, float(pos_w)])

    not_cols = ["acct","from_acct","to_acct"]
    features_cat= [col for col in X_train.columns if col not in not_cols+['Month','Year','year_month']]


    clf = cat_model
    X = X_train[features_cat]
    test = X_test[features_cat]

    folds = KFold(n_splits=10, shuffle=True, random_state=2025)
    oofs  = np.zeros((len(X)))
    test_predictions = np.zeros((len(test)))


    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y_train)):
        X_trn, y_trn = X.iloc[trn_idx], y_train.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y_train.iloc[val_idx]

    
        clf.fit(X_trn, y_trn, eval_set = [(X_val, y_val)])

        vp = clf.predict(X_val)
        oofs[val_idx] = vp
        val_score = f1_score((vp), (y_val))
        print(4*'-- -- -- --')
        print(f'Fold {fold_+1} Val score: {val_score}')
        print(4*'-- -- -- --')

        tp = clf.predict(test)
        test_predictions += tp / folds.n_splits
        
    print(len(test_predictions))
    print()
    print(3*'###',10*"^",3*'###')
    print(f1_score(y_train, oofs))
    
    out_path = "result.csv"
    rounded = [round(x) for x in test_predictions]
    OutputCSV(out_path, df_test, X_test, rounded)
    
