"""
╔══════════════════════════════════════════════════════════════════════╗
║   PARAMETER OPTIMIZER PRO v2                                       ║
║   Multi-timeframe + Multi-indicador + Otimização de parâmetros     ║
║   Baseado nas 3 melhores estratégias CHoCH encontradas             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys, os, json, time, warnings
sys.path.insert(0, "/workspace")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from itertools import product
from joblib import Parallel, delayed

# ══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════

CSV_1MIN  = "/workspace/strategy_composer/wdo_clean.csv"
CSV_5MIN  = "/workspace/strategy_composer/wdo_5min.csv"
OUT_DIR   = "/workspace/param_opt_output"
CAPITAL   = 50_000.0
MULT_WDO  = 10.0
CONTRATOS = 2
COMISSAO  = 5.0   # por contrato por lado
SLIP      = 1.0   # pontos

N_CORES   = min(32, os.cpu_count() or 4)

# Filtros mínimos
MIN_TRADES = 100
MIN_PF     = 1.3
MAX_DD     = -20.0
MIN_WR     = 35.0

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CARREGAR E PREPARAR DADOS MULTI-TIMEFRAME
# ══════════════════════════════════════════════════════════════════════

def carregar_multitf(csv_1min, csv_5min):
    """Carrega e prepara dados em 1min, 5min, 15min, 30min"""
    print("[DATA] Carregando dados multi-timeframe...")

    # 1 minuto
    df1 = pd.read_csv(csv_1min, parse_dates=["datetime"], index_col="datetime")
    df1.columns = [c.lower() for c in df1.columns]
    df1 = df1[df1.index.dayofweek < 5]
    df1 = df1[(df1.index.hour >= 9) & (df1.index.hour < 18)]
    df1 = df1[~df1.index.duplicated(keep="last")].sort_index().dropna()
    df1 = df1[df1["close"] > 0]

    # 5 minutos
    df5 = pd.read_csv(csv_5min, parse_dates=["datetime"], index_col="datetime")
    df5.columns = [c.lower() for c in df5.columns]
    df5 = df5[df5.index.dayofweek < 5]
    df5 = df5[(df5.index.hour >= 9) & (df5.index.hour < 18)]
    df5 = df5[~df5.index.duplicated(keep="last")].sort_index().dropna()

    # 15 minutos (resample do 5min)
    df15 = df5.resample("15min").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df15 = df15[df15.index.dayofweek < 5]
    df15 = df15[(df15.index.hour >= 9) & (df15.index.hour < 18)]

    # 30 minutos
    df30 = df5.resample("30min").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    df30 = df30[df30.index.dayofweek < 5]
    df30 = df30[(df30.index.hour >= 9) & (df30.index.hour < 18)]

    print(f"  1min : {len(df1):,} candles")
    print(f"  5min : {len(df5):,} candles")
    print(f"  15min: {len(df15):,} candles")
    print(f"  30min: {len(df30):,} candles")

    return {"1min": df1, "5min": df5, "15min": df15, "30min": df30}

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — INDICADORES COMPLETOS
# ══════════════════════════════════════════════════════════════════════

def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_sma(s, n): return s.rolling(n).mean()

def calc_atr(df, n=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, fast=12, slow=26, sig=9):
    m = calc_ema(s, fast) - calc_ema(s, slow)
    sg = calc_ema(m, sig)
    return m, sg, m - sg

def calc_adx(df, n=14):
    hi, lo, cl = df["high"], df["low"], df["close"]
    up = hi.diff(); dn = -lo.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=n, adjust=False).mean()
    pdi = pd.Series(pdm, index=df.index).ewm(span=n, adjust=False).mean() / atr * 100
    ndi = pd.Series(ndm, index=df.index).ewm(span=n, adjust=False).mean() / atr * 100
    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100
    return dx.ewm(span=n, adjust=False).mean(), pdi, ndi

def calc_bb(s, n=20, std=2.0):
    m = s.rolling(n).mean()
    d = s.rolling(n).std()
    return m + std*d, m, m - std*d

def calc_stoch(df, k=14, d=3):
    lo_k = df["low"].rolling(k).min()
    hi_k = df["high"].rolling(k).max()
    kp = 100 * (df["close"] - lo_k) / (hi_k - lo_k).replace(0, np.nan)
    return kp, kp.rolling(d).mean()

def calc_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cv = df.groupby(df.index.date)["volume"].cumsum()
    ct = (tp * df["volume"]).groupby(df.index.date).cumsum()
    return ct / cv.replace(0, np.nan)

def calc_supertrend(df, n=10, mult=3.0):
    atr = calc_atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    up  = (hl2 - mult * atr).values
    dn  = (hl2 + mult * atr).values
    cl  = df["close"].values
    dir_v = np.ones(len(df))
    for i in range(1, len(df)):
        up[i] = max(up[i], up[i-1]) if cl[i-1] > up[i-1] else up[i]
        dn[i] = min(dn[i], dn[i-1]) if cl[i-1] < dn[i-1] else dn[i]
        if cl[i] > dn[i-1]:   dir_v[i] = 1
        elif cl[i] < up[i-1]: dir_v[i] = -1
        else:                  dir_v[i] = dir_v[i-1]
    return pd.Series(dir_v, index=df.index)

def calc_choch(df, swing_n=5):
    hi, lo = df["high"], df["low"]
    sh = hi.rolling(swing_n*2+1, center=True).max() == hi
    sl = lo.rolling(swing_n*2+1, center=True).min() == lo
    choch = pd.Series(0, index=df.index)
    last_sh = last_sl = None
    trend = 0
    for i in range(swing_n, len(df)):
        if sh.iloc[i]:
            if trend == -1 and last_sl is not None:
                choch.iloc[i] = 1
            last_sh = hi.iloc[i]
            trend = 1
        if sl.iloc[i]:
            if trend == 1 and last_sh is not None:
                choch.iloc[i] = -1
            last_sl = lo.iloc[i]
            trend = -1
    return choch

def preparar_df(df):
    """Prepara todos os indicadores num dataframe"""
    d = df.copy()
    cl = d["close"]

    # ATR múltiplos
    d["atr_7"]   = calc_atr(d, 7)
    d["atr_14"]  = calc_atr(d, 14)
    d["atr_21"]  = calc_atr(d, 21)
    d["atr_slow"]= calc_atr(d, 50)

    # EMAs
    for p in [9, 20, 50, 100, 200]:
        d[f"ema_{p}"] = calc_ema(cl, p)

    # SMAs
    for p in [20, 50]:
        d[f"sma_{p}"] = calc_sma(cl, p)

    # RSI
    for p in [9, 14, 21]:
        d[f"rsi_{p}"] = calc_rsi(cl, p)

    # MACD
    d["macd"], d["macd_sig"], d["macd_hist"] = calc_macd(cl)

    # ADX
    d["adx"], d["pdi"], d["ndi"] = calc_adx(d)

    # Bollinger
    d["bb_up"], d["bb_mid"], d["bb_lo"] = calc_bb(cl)
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / d["bb_mid"]

    # Estocástico
    d["stoch_k"], d["stoch_d"] = calc_stoch(d)

    # VWAP
    d["vwap"] = calc_vwap(d)

    # Supertrend
    d["supertrend"] = calc_supertrend(d)

    # Volume
    d["vol_ma_10"] = d["volume"].rolling(10).mean()
    d["vol_ma_20"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol_ma_20"].replace(0, np.nan)

    # ROC
    d["roc_5"]  = cl.pct_change(5) * 100
    d["roc_10"] = cl.pct_change(10) * 100

    # CHoCH com swing_n padrão (será sobrescrito pelo otimizador)
    d["choch_5"]  = calc_choch(d, 5)
    d["choch_7"]  = calc_choch(d, 7)
    d["choch_10"] = calc_choch(d, 10)
    d["choch_15"] = calc_choch(d, 15)

    return d.dropna()

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — GERADOR DE SINAL PARAMETRIZADO
# ══════════════════════════════════════════════════════════════════════

def gerar_sinal_parametrizado(df, params, df_htf=None):
    """
    Gera sinal CHoCH com todos os parâmetros otimizáveis.
    df_htf = dataframe de timeframe maior para filtros de tendência
    """
    cl = df["close"]
    n  = len(df)
    true_s  = pd.Series(True,  index=df.index)
    false_s = pd.Series(False, index=df.index)

    # ── SINAL BASE: CHoCH invertido ──────────────────────────────────
    swing_n = params.get("swing_n", 5)
    choch_col = f"choch_{swing_n}" if f"choch_{swing_n}" in df.columns else "choch_5"
    choch = df[choch_col]
    # Invertido: bull quando choch == -1, bear quando choch == 1
    bull_base = choch == -1
    bear_base = choch == 1

    # ── FILTRO 1: RSI ─────────────────────────────────────────────────
    rsi_periodo = params.get("rsi_periodo", 14)
    rsi_limite  = params.get("rsi_limite", 0)  # 0 = desligado
    rsi_col = f"rsi_{rsi_periodo}" if f"rsi_{rsi_periodo}" in df.columns else "rsi_14"

    if rsi_limite > 0:
        # Para compra: RSI deve estar abaixo do limite (oversold)
        # Para venda: RSI deve estar acima de 100-limite (overbought)
        rsi_bull = df[rsi_col] < rsi_limite
        rsi_bear = df[rsi_col] > (100 - rsi_limite)
    else:
        rsi_bull = rsi_bear = true_s

    # ── FILTRO 2: ADX (força da tendência) ───────────────────────────
    adx_min = params.get("adx_min", 0)
    if adx_min > 0:
        adx_ok = df["adx"] >= adx_min
    else:
        adx_ok = true_s

    # ── FILTRO 3: Volume ──────────────────────────────────────────────
    vol_min = params.get("vol_min", 0.0)
    if vol_min > 0:
        vol_ok = df["vol_ratio"] >= vol_min
    else:
        vol_ok = true_s

    # ── FILTRO 4: ATR (evita mercado muito volátil ou muito calmo) ────
    atr_lo = params.get("atr_lo", 0.0)
    atr_hi = params.get("atr_hi", 999.0)
    if atr_lo > 0 or atr_hi < 999:
        atr_ok = (df["atr_14"] >= atr_lo) & (df["atr_14"] <= atr_hi)
    else:
        atr_ok = true_s

    # ── FILTRO 5: Bollinger Band (evita squeeze) ──────────────────────
    bb_min_width = params.get("bb_min_width", 0.0)
    if bb_min_width > 0:
        bb_ok = df["bb_width"] >= bb_min_width
    else:
        bb_ok = true_s

    # ── FILTRO 6: MACD histograma ─────────────────────────────────────
    usar_macd = params.get("usar_macd", False)
    if usar_macd:
        macd_bull = df["macd_hist"] > 0
        macd_bear = df["macd_hist"] < 0
    else:
        macd_bull = macd_bear = true_s

    # ── FILTRO 7: EMA tendência ───────────────────────────────────────
    ema_filtro = params.get("ema_filtro", "NENHUM")
    if ema_filtro == "EMA_20_50":
        ema_bull = df["ema_20"] > df["ema_50"]
        ema_bear = df["ema_20"] < df["ema_50"]
    elif ema_filtro == "EMA_50_200":
        ema_bull = df["ema_50"] > df["ema_200"]
        ema_bear = df["ema_50"] < df["ema_200"]
    elif ema_filtro == "EMA_20_200":
        ema_bull = df["ema_20"] > df["ema_200"]
        ema_bear = df["ema_20"] < df["ema_200"]
    elif ema_filtro == "VWAP":
        ema_bull = cl > df["vwap"]
        ema_bear = cl < df["vwap"]
    elif ema_filtro == "SUPERTREND":
        ema_bull = df["supertrend"] == 1
        ema_bear = df["supertrend"] == -1
    else:
        ema_bull = ema_bear = true_s

    # ── FILTRO 8: Estocástico ─────────────────────────────────────────
    stoch_filtro = params.get("stoch_filtro", False)
    if stoch_filtro:
        stoch_bull = df["stoch_k"] < 50
        stoch_bear = df["stoch_k"] > 50
    else:
        stoch_bull = stoch_bear = true_s

    # ── FILTRO 9: Tendência higher timeframe ──────────────────────────
    htf_filtro = params.get("htf_filtro", "NENHUM")
    if df_htf is not None and htf_filtro != "NENHUM":
        # Reindexar HTF para o mesmo índice do df principal
        htf_prep = df_htf.reindex(df.index, method="ffill")
        if htf_filtro == "EMA_20_50_HTF":
            if "ema_20" in htf_prep.columns and "ema_50" in htf_prep.columns:
                htf_bull = htf_prep["ema_20"] > htf_prep["ema_50"]
                htf_bear = htf_prep["ema_20"] < htf_prep["ema_50"]
            else:
                htf_bull = htf_bear = true_s
        elif htf_filtro == "SUPERTREND_HTF":
            if "supertrend" in htf_prep.columns:
                htf_bull = htf_prep["supertrend"] == 1
                htf_bear = htf_prep["supertrend"] == -1
            else:
                htf_bull = htf_bear = true_s
        elif htf_filtro == "MACD_HTF":
            if "macd_hist" in htf_prep.columns:
                htf_bull = htf_prep["macd_hist"] > 0
                htf_bear = htf_prep["macd_hist"] < 0
            else:
                htf_bull = htf_bear = true_s
        else:
            htf_bull = htf_bear = true_s
    else:
        htf_bull = htf_bear = true_s

    # ── SESSÃO ────────────────────────────────────────────────────────
    hora_min = params.get("hora_min", 9)
    hora_max = params.get("hora_max", 18)
    hour = df.index.hour
    sessao_ok = (hour >= hora_min) & (hour < hora_max)

    # ── COMBINA TUDO ──────────────────────────────────────────────────
    sinal = pd.Series(0, index=df.index)

    mask_bull = (bull_base & rsi_bull & adx_ok & vol_ok & atr_ok &
                 bb_ok & macd_bull & ema_bull & stoch_bull & htf_bull & sessao_ok)
    mask_bear = (bear_base & rsi_bear & adx_ok & vol_ok & atr_ok &
                 bb_ok & macd_bear & ema_bear & stoch_bear & htf_bear & sessao_ok)

    sinal[mask_bull] = 1
    sinal[mask_bear] = -1

    return sinal

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — BACKTEST NUMPY VETORIZADO
# ══════════════════════════════════════════════════════════════════════

def backtest_numpy(hi, lo, cl, sig, stop_pts, win_pts):
    n = len(cl)
    capital  = CAPITAL
    pnl_arr  = np.zeros(n)
    res_arr  = np.zeros(n, dtype=np.int8)
    em_pos   = False
    entry = sl = tp = 0.0
    direcao = 0
    n_trades = n_wins = 0

    for i in range(1, n):
        if em_pos:
            hit_sl = (direcao == 1 and lo[i] <= sl) or (direcao == -1 and hi[i] >= sl)
            hit_tp = (direcao == 1 and hi[i] >= tp) or (direcao == -1 and lo[i] <= tp)
            if hit_sl or hit_tp:
                saida = tp if hit_tp else sl
                pts   = (saida - entry) * direcao
                brl   = pts * MULT_WDO * CONTRATOS - COMISSAO * CONTRATOS * 2
                capital += brl
                pnl_arr[i] = brl
                res_arr[i] = 1 if hit_tp else -1
                n_trades += 1
                if hit_tp: n_wins += 1
                em_pos = False
            continue

        if sig[i] != 0:
            direcao = int(sig[i])
            entry   = cl[i] + SLIP * direcao
            sl      = entry - direcao * stop_pts
            tp      = entry + direcao * win_pts
            em_pos  = True

    return pnl_arr, res_arr, n_trades, n_wins

def calcular_metricas(pnl_arr, res_arr, n_trades, n_wins):
    if n_trades < MIN_TRADES:
        return None

    n_losses = n_trades - n_wins
    wr = n_wins / n_trades * 100
    if wr < MIN_WR:
        return None

    pnl_w = pnl_arr[res_arr == 1].sum()
    pnl_l = pnl_arr[res_arr == -1].sum()
    pf    = abs(pnl_w / pnl_l) if pnl_l != 0 else 9999.0
    if pf < MIN_PF:
        return None

    total_pnl = pnl_arr.sum()
    capital   = CAPITAL + np.cumsum(pnl_arr)
    peak      = np.maximum.accumulate(capital)
    dd        = (capital - peak) / peak * 100
    mdd       = dd.min()
    if mdd < MAX_DD:
        return None

    rets    = pd.Series(capital).pct_change().dropna()
    sharpe  = rets.mean() / rets.std() * np.sqrt(252*108) if rets.std() > 0 else 0.0
    neg     = rets[rets < 0]
    sortino = rets.mean() / neg.std() * np.sqrt(252*108) if len(neg) > 0 else 0.0

    avg_w = pnl_arr[res_arr==1].mean() / (MULT_WDO*CONTRATOS) if n_wins > 0 else 0
    avg_l = pnl_arr[res_arr==-1].mean() / (MULT_WDO*CONTRATOS) if n_losses > 0 else 0
    exp   = (wr/100)*avg_w + ((1-wr/100)*avg_l)

    return {
        "total_trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "avg_win_pts": round(avg_w, 2),
        "avg_loss_pts": round(avg_l, 2),
        "expectancia_pts": round(exp, 2),
        "total_pnl_brl": round(total_pnl, 2),
        "retorno_pct": round(total_pnl / CAPITAL * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final": round(CAPITAL + total_pnl, 2),
    }

def calcular_score(m):
    if not m: return -999.0
    pf      = min(m["profit_factor"], 10)
    sharpe  = min(max(m["sharpe"], 0), 8)
    sortino = min(max(m["sortino"], 0), 10)
    wr      = m["win_rate"] / 100
    trades  = min(m["total_trades"], 1000)
    dd      = abs(m["max_drawdown_pct"])
    ret     = min(max(m["retorno_pct"], -100), 300) / 300
    exp     = min(max(m["expectancia_pts"], -10), 20) / 20
    return round(
        pf/10       * 0.20
        + sharpe/8  * 0.15
        + sortino/10* 0.10
        + wr        * 0.15
        + trades/1000*0.10
        + ret       * 0.10
        + exp       * 0.15
        - dd/20     * 0.05
    , 6)

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — ESPAÇO DE PARÂMETROS
# ══════════════════════════════════════════════════════════════════════

PARAM_SPACE = {
    # Stop e Win em pontos
    "stop_pts":      [3, 4, 5, 6, 7, 8, 10],
    "win_pts":       [9, 12, 15, 18, 20, 21, 25, 28, 30],

    # CHoCH: janela do swing
    "swing_n":       [5, 7, 10, 15],

    # Sessão: horário de entrada
    "hora_min":      [9, 10, 11],
    "hora_max":      [13, 15, 17, 18],

    # Filtros técnicos
    "rsi_periodo":   [9, 14, 21],
    "rsi_limite":    [0, 30, 35, 40],     # 0=desligado

    "adx_min":       [0, 15, 20, 25],     # 0=desligado
    "vol_min":       [0.0, 0.8, 1.0, 1.5],
    "atr_lo":        [0.0, 2.0, 3.0],
    "atr_hi":        [999.0, 15.0, 20.0],
    "bb_min_width":  [0.0, 0.003, 0.005],

    # Filtros direcionais
    "usar_macd":     [False, True],
    "stoch_filtro":  [False, True],
    "ema_filtro":    ["NENHUM", "EMA_20_50", "EMA_50_200", "VWAP", "SUPERTREND"],

    # Higher timeframe
    "htf_timeframe": ["NENHUM", "15min", "30min"],
    "htf_filtro":    ["NENHUM", "EMA_20_50_HTF", "SUPERTREND_HTF", "MACD_HTF"],
}

def gerar_combinacoes_aleatorias(n=5000, seed=42):
    """Gera N combinações aleatórias do espaço de parâmetros"""
    rng = np.random.default_rng(seed)
    combos = []
    for _ in range(n):
        p = {}
        for k, v in PARAM_SPACE.items():
            p[k] = v[rng.integers(len(v))]
        # Garante win > stop
        while p["win_pts"] <= p["stop_pts"]:
            p["win_pts"] = PARAM_SPACE["win_pts"][rng.integers(len(PARAM_SPACE["win_pts"]))]
        # Garante hora_min < hora_max
        while p["hora_min"] >= p["hora_max"]:
            p["hora_max"] = PARAM_SPACE["hora_max"][rng.integers(len(PARAM_SPACE["hora_max"]))]
        # htf_filtro só faz sentido se htf_timeframe != NENHUM
        if p["htf_timeframe"] == "NENHUM":
            p["htf_filtro"] = "NENHUM"
        combos.append(p)
    return combos

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — OTIMIZAÇÃO PARALELA
# ══════════════════════════════════════════════════════════════════════

_DFS_GLOBAL = None

def _init_worker(dfs):
    global _DFS_GLOBAL
    _DFS_GLOBAL = dfs

def _testar_params(args):
    estrategia_nome, params = args
    try:
        dfs = _DFS_GLOBAL
        df5  = dfs["5min_ins"]
        htf_tf = params.get("htf_timeframe", "NENHUM")
        df_htf = dfs.get(f"{htf_tf}_ins") if htf_tf != "NENHUM" else None

        sinal = gerar_sinal_parametrizado(df5, params, df_htf)
        n_sig = (sinal != 0).sum()
        if n_sig < MIN_TRADES * 0.3:
            return None

        pnl, res, nt, nw = backtest_numpy(
            df5["high"].values, df5["low"].values,
            df5["close"].values, sinal.values,
            params["stop_pts"], params["win_pts"]
        )
        m = calcular_metricas(pnl, res, nt, nw)
        if not m:
            return None

        s = calcular_score(m)
        return {
            "score": s,
            "estrategia": estrategia_nome,
            "params": params,
            **m,
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════

def walk_forward(dfs_full, melhor, n_splits=5):
    print(f"\n[WF] Walk-Forward {n_splits} splits...")
    params = melhor["params"]
    df5 = dfs_full["5min"]
    step = len(df5) // n_splits
    splits = []

    for i in range(n_splits - 1):
        inicio = i * step
        fim    = (i + 2) * step
        split  = inicio + int((fim - inicio) * 0.7)

        df_tr = df5.iloc[inicio:split]
        df_te = df5.iloc[split:fim]
        if len(df_tr) < 500 or len(df_te) < 200:
            continue

        resultados = []
        for d in [df_tr, df_te]:
            try:
                htf_tf = params.get("htf_timeframe", "NENHUM")
                df_htf = None
                if htf_tf != "NENHUM":
                    freq = "15min" if htf_tf == "15min" else "30min"
                    df_htf_raw = d.resample(freq).agg({
                        "open":"first","high":"max","low":"min",
                        "close":"last","volume":"sum"
                    }).dropna()
                    df_htf = preparar_df(df_htf_raw)

                d_prep = preparar_df(d)
                sig = gerar_sinal_parametrizado(d_prep, params, df_htf)
                pnl, res, nt, nw = backtest_numpy(
                    d_prep["high"].values, d_prep["low"].values,
                    d_prep["close"].values, sig.values,
                    params["stop_pts"], params["win_pts"]
                )
                m = calcular_metricas(pnl, res, nt, nw)
                s = calcular_score(m)
            except Exception as e:
                m, s = None, -999.0
            resultados.append((s, m))

        s_tr, m_tr = resultados[0]
        s_te, m_te = resultados[1]

        lucro_te = m_te.get("total_pnl_brl", 0) if m_te else 0
        emoji = "✅" if lucro_te > 0 else "❌"

        print(f"  Split {i+1}: TRAIN score={s_tr:.4f} WR={m_tr.get('win_rate',0) if m_tr else 0:.1f}% "
              f"trades={m_tr.get('total_trades',0) if m_tr else 0} "
              f"PnL=R${m_tr.get('total_pnl_brl',0) if m_tr else 0:,.0f} | "
              f"TEST score={s_te:.4f} WR={m_te.get('win_rate',0) if m_te else 0:.1f}% "
              f"trades={m_te.get('total_trades',0) if m_te else 0} "
              f"PnL=R${lucro_te:,.0f} {emoji}")

        splits.append({
            "split": i+1,
            "score_train": s_tr, "score_test": s_te,
            "train": {k:v for k,v in (m_tr or {}).items()},
            "test":  {k:v for k,v in (m_te or {}).items()},
        })

    scores_oos = [s["score_test"] for s in splits if s["score_test"] > 0]
    lucrativos = [s for s in splits if s.get("test",{}).get("total_pnl_brl",0) > 0]

    return {
        "splits": splits,
        "wf_score_medio": round(np.mean(scores_oos), 4) if scores_oos else -999,
        "splits_lucrativos": len(lucrativos),
        "total_splits": len(splits),
    }

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — RELATÓRIO
# ══════════════════════════════════════════════════════════════════════

def exibir_relatorio(resultados, wf, m_oos):
    print(f"\n{'═'*90}")
    print("  PARAMETER OPTIMIZER PRO — RESULTADO FINAL")
    print(f"{'═'*90}\n")

    print(f"  TOP 15 CONFIGURAÇÕES\n")
    print(f"  {'#':>2} {'Estratégia':<22} {'SL':>3} {'TP':>3} {'Swing':>5} "
          f"{'HorMin':>6} {'HorMax':>6} {'EMA':>12} {'HTF':>10} "
          f"{'Score':>7} {'PF':>6} {'WR%':>6} {'Trades':>7} {'PnL R$':>10}")
    print(f"  {'-'*120}")

    for i, r in enumerate(resultados[:15], 1):
        p = r["params"]
        print(f"  {i:>2} {r['estrategia']:<22} "
              f"{p['stop_pts']:>3} {p['win_pts']:>3} {p['swing_n']:>5} "
              f"{p['hora_min']:>6} {p['hora_max']:>6} "
              f"{p['ema_filtro']:>12} {p.get('htf_filtro','NENHUM'):>10} "
              f"{r['score']:>7.4f} {r['profit_factor']:>6.3f} "
              f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
              f"{r['total_pnl_brl']:>10,.0f}")

    if resultados:
        best = resultados[0]
        p = best["params"]
        print(f"\n{'═'*90}")
        print("  ★ MELHOR CONFIGURAÇÃO COMPLETA")
        print(f"{'═'*90}")
        print(f"  Estratégia base  : {best['estrategia']}")
        print(f"\n  PARÂMETROS DE ENTRADA:")
        print(f"    Stop Loss      : {p['stop_pts']} pontos = R${p['stop_pts']*MULT_WDO*CONTRATOS:.0f}/trade")
        print(f"    Take Profit    : {p['win_pts']} pontos = R${p['win_pts']*MULT_WDO*CONTRATOS:.0f}/trade")
        print(f"    Risk/Reward    : 1:{p['win_pts']/p['stop_pts']:.1f}")
        print(f"    Swing Length   : {p['swing_n']} barras")
        print(f"    Horário        : {p['hora_min']}h às {p['hora_max']}h")
        print(f"\n  FILTROS ATIVOS:")
        print(f"    EMA            : {p['ema_filtro']}")
        print(f"    MACD Hist      : {'Sim' if p.get('usar_macd') else 'Não'}")
        print(f"    RSI            : período {p['rsi_periodo']} | limite {p['rsi_limite']} {'(desligado)' if p['rsi_limite']==0 else ''}")
        print(f"    ADX mínimo     : {p['adx_min']} {'(desligado)' if p['adx_min']==0 else ''}")
        print(f"    Volume mínimo  : {p['vol_min']}x média {'(desligado)' if p['vol_min']==0 else ''}")
        print(f"    ATR range      : {p['atr_lo']}-{p['atr_hi']} pts")
        print(f"    BB width min   : {p['bb_min_width']} {'(desligado)' if p['bb_min_width']==0 else ''}")
        print(f"    Estocástico    : {'Sim' if p.get('stoch_filtro') else 'Não'}")
        print(f"    HTF timeframe  : {p.get('htf_timeframe','NENHUM')}")
        print(f"    HTF filtro     : {p.get('htf_filtro','NENHUM')}")
        print(f"\n  RESULTADO IN-SAMPLE:")
        print(f"    Score          : {best['score']:.4f}")
        print(f"    Profit Factor  : {best['profit_factor']}")
        print(f"    Win Rate       : {best['win_rate']}%")
        print(f"    Total Trades   : {best['total_trades']}")
        print(f"    Expectância    : {best['expectancia_pts']:.2f} pts/trade")
        print(f"    Sharpe         : {best['sharpe']}")
        print(f"    Max Drawdown   : {best['max_drawdown_pct']}%")
        print(f"    Retorno        : {best['retorno_pct']}%")
        print(f"    Capital Final  : R${best['capital_final']:,.2f}")

    print(f"\n{'═'*90}")
    print("  WALK-FORWARD VALIDATION")
    print(f"{'═'*90}")
    print(f"  Score OOS médio  : {wf['wf_score_medio']}")
    print(f"  Splits lucrativos: {wf['splits_lucrativos']}/{wf['total_splits']}")

    if m_oos:
        print(f"\n{'═'*90}")
        print("  OUT-OF-SAMPLE (dados nunca vistos)")
        print(f"{'═'*90}")
        print(f"  Trades         : {m_oos['total_trades']}")
        print(f"  Win Rate       : {m_oos['win_rate']}%")
        print(f"  Profit Factor  : {m_oos['profit_factor']}")
        print(f"  Sharpe         : {m_oos['sharpe']}")
        print(f"  Max Drawdown   : {m_oos['max_drawdown_pct']}%")
        print(f"  Retorno        : {m_oos['retorno_pct']}%")
        print(f"  Capital Final  : R${m_oos['capital_final']:,.2f}")

def salvar(resultados, wf, m_oos):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [{k:v for k,v in r.items() if k != "equity"} for r in resultados[:100]]
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/param_ranking.csv", index=False)
    dashboard = {
        "melhor": {k:v for k,v in resultados[0].items()} if resultados else {},
        "top10": resultados[:10],
        "walk_forward": wf,
        "oos": m_oos,
        "gerado_em": pd.Timestamp.now().isoformat(),
    }
    with open(f"{OUT_DIR}/param_resultado.json", "w") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OUT] Salvo em {OUT_DIR}/")

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("╔" + "═"*72 + "╗")
    print("║  PARAMETER OPTIMIZER PRO v2                                           ║")
    print("║  CHoCH invertido | Multi-timeframe | 20+ parâmetros | 5.000 combos   ║")
    print("╚" + "═"*72 + "╝\n")

    # 1. Carregar dados
    dfs_raw = carregar_multitf(CSV_1MIN, CSV_5MIN)

    # 2. Split in-sample / out-of-sample
    df5 = dfs_raw["5min"]
    split = int(len(df5) * 0.70)

    print("\n[1/5] Preparando indicadores em todos os timeframes...")
    dfs_ins  = {}
    dfs_full = {}

    for tf, df in dfs_raw.items():
        sp = int(len(df) * 0.70)
        dfs_ins[f"{tf}_ins"]  = preparar_df(df.iloc[:sp])
        dfs_full[tf]          = preparar_df(df)
        print(f"  {tf}: {len(dfs_ins[f'{tf}_ins']):,} candles in-sample | {len(dfs_full[tf]):,} total")

    # 3. Gerar combinações
    print("\n[2/5] Gerando combinações de parâmetros...")
    combos = gerar_combinacoes_aleatorias(n=5000, seed=42)
    print(f"  {len(combos):,} combinações geradas")

    # Distribui entre as 3 estratégias base
    estrategias_base = [
        "CHoCH_DIA_INTEIRO",
        "CHoCH_MACD_SEM_ALMOCO",
        "CHoCH_EMA_MANHA",
    ]
    args = []
    for i, params in enumerate(combos):
        est = estrategias_base[i % len(estrategias_base)]
        args.append((est, params))

    print(f"  {len(args):,} testes ({N_CORES} cores)\n")

    # 4. Otimização paralela
    print("[3/5] Otimizando parâmetros...")
    results_raw = Parallel(
        n_jobs=N_CORES, backend="loky", verbose=5,
        initializer=_init_worker, initargs=(dfs_ins,)
    )(delayed(_testar_params)(a) for a in args)

    resultados = [r for r in results_raw if r is not None]
    resultados.sort(key=lambda x: -x["score"])
    print(f"\n  {len(resultados)} válidos de {len(args):,}")

    if not resultados:
        print("\n⚠ Nenhuma configuração válida.")
        print(f"  Filtros atuais: MIN_TRADES={MIN_TRADES}, MIN_PF={MIN_PF}, MAX_DD={MAX_DD}%")
        return

    # 5. Walk-Forward
    print("\n[4/5] Walk-Forward Validation...")
    wf = walk_forward(dfs_full, resultados[0])

    # 6. Out-of-Sample
    print("\n[5/5] Out-of-Sample...")
    best = resultados[0]
    p = best["params"]
    df5_oos = dfs_full["5min"].iloc[split:]

    try:
        htf_tf = p.get("htf_timeframe", "NENHUM")
        df_htf_oos = None
        if htf_tf != "NENHUM":
            freq = "15min" if htf_tf == "15min" else "30min"
            df_htf_raw = df5_oos.resample(freq).agg({
                "open":"first","high":"max","low":"min",
                "close":"last","volume":"sum"
            }).dropna()
            df_htf_oos = preparar_df(df_htf_raw)

        sig_oos = gerar_sinal_parametrizado(df5_oos, p, df_htf_oos)
        pnl_o, res_o, nt_o, nw_o = backtest_numpy(
            df5_oos["high"].values, df5_oos["low"].values,
            df5_oos["close"].values, sig_oos.values,
            p["stop_pts"], p["win_pts"]
        )
        m_oos = calcular_metricas(pnl_o, res_o, nt_o, nw_o)
        if m_oos:
            print(f"  OOS: PF={m_oos['profit_factor']} WR={m_oos['win_rate']}% "
                  f"Trades={m_oos['total_trades']} PnL=R${m_oos['total_pnl_brl']:,.0f}")
        else:
            print("  OOS: não passou nos filtros")
    except Exception as e:
        print(f"  OOS erro: {e}")
        m_oos = None

    # 7. Salvar e exibir
    salvar(resultados, wf, m_oos)
    exibir_relatorio(resultados, wf, m_oos)

    elapsed = time.time() - t0
    print(f"\n  Tempo total: {elapsed/60:.1f} minutos")
    print(f"  Válidos: {len(resultados)} | Outputs: {OUT_DIR}/")


if __name__ == "__main__":
    main()
