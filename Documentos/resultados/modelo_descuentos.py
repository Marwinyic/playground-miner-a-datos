"""
Playground - Modelo predictivo: ¿habrá descuento en los próximos 30 días?
Dataset: Steam Games Dataset (Wannigamage et al., Mendeley Data, DOI 10.17632/ycy3sy3vj2.1)
Uso:  python modelo_descuentos.py PriceHistory/   (carpeta con los CSV por appid)
"""
import sys, glob, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

HORIZON = 30
STEP = 7                      # una observación por juego cada 7 días (reduce autocorrelación)
TRAIN_END = "2019-12-01"      # train: t <= esta fecha (su ventana objetivo termina antes de 2020)
TEST_START = "2020-01-01"     # test: t >= esta fecha (separación temporal, sin solapamiento)
SEED = 42

def build_app(df, appid):
    df = df.sort_values("Date").set_index("Date")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max()))
    df[["Initialprice", "Finalprice"]] = df[["Initialprice", "Finalprice"]].ffill()
    df["Discount"] = df["Discount"].fillna(0)
    disc = (df["Discount"] > 0).astype(int)
    out = pd.DataFrame(index=df.index)
    # --- variable objetivo: algún descuento en t+1 ... t+30
    out["y"] = disc[::-1].rolling(HORIZON, min_periods=HORIZON).sum()[::-1].shift(-1)
    out["y"] = (out["y"] > 0).astype(float).where(out["y"].notna())
    # --- predictoras (solo información hasta t)
    out["discount_now"] = df["Discount"]
    out["on_sale_now"] = disc
    out["initial_price"] = df["Initialprice"]
    out["price_ratio"] = (df["Finalprice"] / df["Initialprice"].replace(0, np.nan)).fillna(1)
    last = df.index.to_series().where(disc == 1).ffill()
    ds = (df.index.to_series() - last).dt.days
    out["no_prev_discount"] = ds.isna().astype(int)
    out["days_since_last_disc"] = ds.fillna(999)
    out["n_disc_days_90"] = disc.rolling(90, min_periods=1).sum()
    out["share_disc_hist"] = disc.expanding().mean()
    out["max_disc_hist"] = df["Discount"].cummax()
    out["min_price_ratio_hist"] = out["price_ratio"].cummin()
    starts = ((disc == 1) & (disc.shift(1, fill_value=0) == 0)).astype(int)
    out["n_episodes_hist"] = starts.cumsum()
    out["days_observed"] = np.arange(len(df))
    out["month"] = df.index.month
    out["appid"] = appid
    out["date"] = df.index
    return out

def main(folder):
    frames = []
    for f in glob.glob(os.path.join(folder, "*.csv")):
        d = pd.read_csv(f, parse_dates=["Date"])
        if len(d) < 60: continue
        frames.append(build_app(d, int(os.path.basename(f)[:-4])))
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["y"])
    data = data[(data.days_observed >= 30)]                    # al menos 30 días de historial
    data = data[(data.date - pd.Timestamp("2019-04-07")).dt.days % STEP == 0]
    feats = [c for c in data.columns if c not in ("y", "appid", "date", "days_observed")]  # days_observed excluida: equivale a la fecha calendario

    tr = data[data.date <= TRAIN_END]; te = data[data.date >= TEST_START]
    Xtr, ytr, Xte, yte = tr[feats], tr.y.astype(int), te[feats], te.y.astype(int)
    res = {"n_train": len(tr), "n_test": len(te), "games": int(data.appid.nunique()),
           "pos_rate_train": float(ytr.mean()), "pos_rate_test": float(yte.mean())}

    def evaluate(name, pred, proba=None):
        r = {"precision": precision_score(yte, pred, zero_division=0),
             "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
             "confusion[[TN,FP],[FN,TP]]": confusion_matrix(yte, pred).tolist()}
        if proba is not None: r["roc_auc"] = roc_auc_score(yte, proba)
        res[name] = r

    # Modelos de referencia (sin aprendizaje)
    evaluate("base_siempre_si", np.ones(len(yte), dtype=int))
    evaluate("base_en_oferta_ahora", te.on_sale_now.values)
    evaluate("base_descuento_ultimos_90d", (te.n_disc_days_90 > 0).astype(int).values)
    # Modelos
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    lr.fit(Xtr, ytr); evaluate("regresion_logistica", lr.predict(Xte), lr.predict_proba(Xte)[:, 1])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, ytr); evaluate("random_forest", rf.predict(Xte), rf.predict_proba(Xte)[:, 1])
    res["importancias_rf"] = dict(sorted(zip(feats, map(float, rf.feature_importances_)), key=lambda x: -x[1]))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    json.dump(res, open("resultados.json", "w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PriceHistory")
