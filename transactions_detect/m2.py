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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

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
    out = pd.DataFrame({
        f'{prefix}_sum': g.sum(),
        f'{prefix}_mean': g.mean(),
        f'{prefix}_std': g.std().fillna(0.0),
        f'{prefix}_min': g.min(),
        f'{prefix}_max': g.max(),
        f'{prefix}_cnt': g.count()
    }).reset_index().rename(columns={key: 'acct'})
    return out


import numpy as np
import pandas as pd

def safe_div(numer, denom, default=0.0):
    numer = np.asarray(numer, dtype="float64")
    denom = np.asarray(denom, dtype="float64")
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.divide(numer, denom, where=(denom != 0))
    # fill invalids with default
    out[~np.isfinite(out)] = default
    return out

def clean_numeric_df(df: pd.DataFrame, clip_quantile: float = 0.999) -> pd.DataFrame:
    """Replace infs with NaN, clip extreme tails, ensure float64."""
    out = df.copy()
    # replace inf/-inf with NaN — SimpleImputer will handle NaN later
    out = out.replace([np.inf, -np.inf], np.nan)
    # clip numeric columns to reduce 'too large for float64' risk from outliers
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        hi = out[num_cols].quantile(clip_quantile).astype(float)
        lo = out[num_cols].quantile(1.0 - clip_quantile).astype(float)
        out[num_cols] = out[num_cols].clip(lower=lo, upper=hi, axis=1)
        out[num_cols] = out[num_cols].astype('float64')
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
    #feats['send_recv_ratio'] = np.where(feats['recv_sum'] > 0, feats['send_sum'] / feats['recv_sum'], np.inf)
    #feats['degree_ratio'] = np.where(feats['in_unique_cnt'] > 0, feats['out_unique_cnt'] / feats['in_unique_cnt'], np.inf)
    #feats['reciprocal_flag'] = ((feats['has_send'] > 0) & (feats['has_recv'] > 0)).astype(int)
    #feats['net_over_gross'] = np.where(feats['gross_amt'] > 0, feats['net_amt'] / feats['gross_amt'], 0.0)

        # SAFE ratios (no infs)
    feats['send_recv_ratio'] = safe_div(feats['send_sum'], feats['recv_sum'], default=0.0)
    feats['degree_ratio']    = safe_div(feats['out_unique_cnt'], feats['in_unique_cnt'], default=0.0)
    feats['reciprocal_flag'] = ((feats['has_send'] > 0) & (feats['has_recv'] > 0)).astype(int)
    feats['net_over_gross']  = safe_div(feats['net_amt'], feats['gross_amt'], default=0.0)

    #feats['total_txn_cnt'] = feats['send_cnt'] + feats['recv_cnt']
    #feats['net_amt'] = feats['send_sum'] - feats['recv_sum']
   # feats['gross_amt'] = feats['send_sum'] + feats['recv_sum']


    
    # Optional: stabilized scales (help trees a bit when distributions are heavy-tailed)
    for col in ['send_sum','recv_sum','gross_amt','out_unique_cnt','in_unique_cnt','total_txn_cnt']:
        feats[f'log1p_{col}'] = np.log1p(feats[col].astype('float64'))


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
    df_feat= df_feat.fillna(0)
    
    df_alert = df_alert.copy()
    df_test = df_test.copy()
    
    df_alert= df_alert.fillna(0)
    df_test = df_test.fillna(0)
    


    # 訓練集：非測試清單 & is_esun==1
    X_train = df_feat[(~df_feat['acct'].isin(df_test['acct'])) & (df_feat['is_esun'] == 1)].copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)

    # 測試集：predict 清單
    X_test = df_feat[df_feat['acct'].isin(df_test['acct'])].copy()

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
    RandomForest + RandomizedSearchCV（較穩健、可平衡失衡）
    - 以 StratifiedKFold=5 做 AUC 與 PR-AUC 監控
    - 使用 class_weight="balanced"（或可改 sample_weight）
    """
    Xtr, feat_cols = _make_features_matrix(X_train)
    Xte, _ = _make_features_matrix(X_test)

    # 簡單補值（樹模型通常不需標準化）
    imputer = SimpleImputer(strategy='median')
    Xtr_ = imputer.fit_transform(Xtr)
    Xte_ = imputer.transform(Xte)

    # 模型與搜尋空間
    base = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=N_JOBS,
        random_state=random_state,
    )

    param_space = {
        'n_estimators': [300, 400, 600, 800],
        'max_depth': [None, 8, 12, 16, 24],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_space,
        n_iter=20,
        scoring='roc_auc',
        cv=cv,
        verbose=1,
        n_jobs=N_JOBS,
        random_state=random_state,
        return_train_score=False,
    )
    search.fit(Xtr_, y_train)

    best_model = search.best_estimator_
    print(f"(CV) Best ROC-AUC: {search.best_score_:.4f} | Best params: {search.best_params_}")

    # 以最佳參數再做 5-fold 檢視 PR-AUC（更對少數類友好）
    oof_pred = np.zeros(len(Xtr_))
    for fold, (tr_idx, va_idx) in enumerate(cv.split(Xtr_, y_train), 1):
        mdl = RandomForestClassifier(**best_model.get_params())
        mdl.fit(Xtr_[tr_idx], y_train.iloc[tr_idx])
        oof_pred[va_idx] = mdl.predict_proba(Xtr_[va_idx])[:, 1]
        roc = roc_auc_score(y_train.iloc[va_idx], oof_pred[va_idx])
        pr = average_precision_score(y_train.iloc[va_idx], oof_pred[va_idx])
        print(f"(CV Fold {fold}) ROC-AUC={roc:.4f}, PR-AUC={pr:.4f}")

    print(f"(OOF) ROC-AUC={roc_auc_score(y_train, oof_pred):.4f}, PR-AUC={average_precision_score(y_train, oof_pred):.4f}")

    # 以全部訓練資料重新訓練
    best_model.fit(Xtr_, y_train)

    # 測試集預測機率
    y_prob = best_model.predict_proba(Xte_)[:, 1]
    # 0/1 標籤（基線先以 0.5 門檻；可考慮以驗證最佳F1或Top-k策略）
    y_pred = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob, feat_cols, best_model, imputer


# =========================
# 5) Output
# =========================
def OutputCSV(path: str, df_test: pd.DataFrame, X_test: pd.DataFrame, y_pred: np.ndarray, y_prob: np.ndarray = None):
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

    if y_prob is not None:
        df_prob = pd.DataFrame({
            'acct': X_test['acct'].values,
            'prob': y_prob,
            'label': y_pred
        })
        prob_path = os.path.splitext(path)[0] + "_with_prob.csv"
        df_prob.to_csv(prob_path, index=False)
        print(f"(Finish) Prob file saved to {prob_path}")


# =========================
# 6) Main
# =========================
if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)

    # 依你的資料所在目錄調整
    dir_path = "data/"

    df_txn, df_alert, df_test = LoadCSV(dir_path)
    df_feat = PreProcessing(df_txn)
    X_train, X_test, y_train = TrainTestSplit(df_feat, df_alert, df_test)

    # 安全檢查：避免空訓練集（若資料特殊）
    if len(X_train) == 0 or y_train.nunique() < 2:
        # 回退到簡單樹（與原baseline類似）— 直接標 0
        print("[WARN] 訓練資料不足或無正負樣本，回退到全 0 預測。")
        y_pred = np.zeros(len(X_test), dtype=int)
        y_prob = np.zeros(len(X_test), dtype=float)
    else:
        y_pred, y_prob, feat_cols, model, imputer = Modeling(X_train, y_train, X_test)

        # 簡單輸出特徵重要度（前20）
        try:
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                imp_df = pd.DataFrame({"feature": feat_cols, "importance": importances}).sort_values("importance", ascending=False)
                print("\n[Top 20 Feature Importances]")
                print(imp_df.head(20).to_string(index=False))
        except Exception as e:
            print(f"[INFO] Skip importance print: {e}")

    out_path = "result.csv"
    OutputCSV(out_path, df_test, X_test, y_pred, y_prob)

