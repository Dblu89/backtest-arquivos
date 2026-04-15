"""
AUTO-DIAGNOSTIC E FIX — Strategy Composer
Testa cada indicador, identifica gargalos e corrige automaticamente.
"""

import warnings
import time
import sys
import os
import importlib

warnings.filterwarnings("ignore")
sys.path.insert(0, "/workspace/strategy_composer")

import numpy as np
import pandas as pd

print("=" * 60)
print(" AUTO-DIAGNOSTIC — Strategy Composer")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. Carregar dados
# ──────────────────────────────────────────────────────────────

print("\n[1/6] Carregando dados...")
df = pd.read_csv(
    "/workspace/wdo_2025.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)
df.columns = [c.lower().strip() for c in df.columns]
df = df[df.index.dayofweek < 5]
df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
df = df.dropna()
df = df[df["close"] > 0]
df = df[~df.index.duplicated(keep="last")]
df = df.sort_index()

# Usar 30k candles para diagnóstico (mais rápido)
df_test = df.iloc[:30000].copy()
print(f"  Total: {len(df):,} candles | Teste: {len(df_test):,} candles")

# ──────────────────────────────────────────────────────────────
# 2. Versões RÁPIDAS de cada indicador (NumPy puro)
# ──────────────────────────────────────────────────────────────

print("\n[2/6] Definindo indicadores otimizados...")


def atr_fast(df, n=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].shift(1).values
    tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
    atr = pd.Series(tr, index=df.index)
    return atr.rolling(n).mean()


def ema_fast(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi_fast(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_fast(s, fast=12, slow=26, sig=9):
    ef = ema_fast(s, fast)
    es = ema_fast(s, slow)
    m = ef - es
    sg = ema_fast(m, sig)
    return m, sg, m - sg


def bollinger_fast(s, n=20, std=2.0):
    ma = s.rolling(n).mean()
    dev = s.rolling(n).std()
    return ma + std * dev, ma, ma - std * dev


def adx_fast(df, n=14):
    h = df["high"].values
    l = df["low"].values
    up = np.diff(h, prepend=h[0])
    down = -np.diff(l, prepend=l[0])
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)

    tr = atr_fast(df, n)
    atr_v = tr.values
    pdm_s = pd.Series(pdm, index=df.index).rolling(n).mean()
    mdm_s = pd.Series(mdm, index=df.index).rolling(n).mean()
    pdi = 100 * pdm_s / (atr_v + 1e-9)
    mdi = 100 * mdm_s / (atr_v + 1e-9)
    denom = (pdi + mdi).replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / denom
    return dx.rolling(n).mean()


def supertrend_fast(df, n=10, mult=3.0):
    """Supertrend com NumPy arrays — sem iloc, sem pandas no loop."""
    atr = atr_fast(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    close = df["close"].values
    sz = len(df)
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    dir_ = np.ones(sz)

    for i in range(1, sz):
        if np.isnan(atr[i]):
            dir_[i] = dir_[i - 1]
            continue
        up[i] = max(up[i], up[i - 1]) if close[i - 1] > up[i - 1] else up[i]
        dn[i] = min(dn[i], dn[i - 1]) if close[i - 1] < dn[i - 1] else dn[i]
        if close[i] > dn[i - 1]:
            dir_[i] = 1
        elif close[i] < up[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]

    return pd.Series(dir_, index=df.index)


def swing_hls_fast(df, n=5):
    """Swing highs/lows com NumPy."""
    h = df["high"].values
    l = df["low"].values
    sz = len(df)
    sh = np.zeros(sz)
    sl = np.zeros(sz)

    for i in range(n, sz - n):
        wh = h[i - n:i + n + 1]
        wl = l[i - n:i + n + 1]
        if h[i] == wh.max() and h[i] > h[i - 1] and h[i] > h[i + 1]:
            sh[i] = h[i]
        if l[i] == wl.min() and l[i] < l[i - 1] and l[i] < l[i + 1]:
            sl[i] = l[i]

    df = df.copy()
    df["sh"] = sh
    df["sl"] = sl
    return df


def bos_choch_fast(df):
    """BOS/CHoCH com arrays NumPy."""
    df = df.copy()
    sh = df["sh"].values
    sl = df["sl"].values
    bos = np.zeros(len(df), dtype=int)
    choch = np.zeros(len(df), dtype=int)
    lsh = lsl = None
    trend = 0

    for i in range(1, len(df)):
        if sh[i] > 0:
            if lsh is not None and sh[i] > lsh:
                if trend == 1:
                    bos[i] = 1
                else:
                    choch[i] = 1
                    trend = 1
            lsh = sh[i]

        if sl[i] > 0:
            if lsl is not None and sl[i] < lsl:
                if trend == -1:
                    bos[i] = -1
                else:
                    choch[i] = -1
                    trend = -1
            lsl = sl[i]

    df["bos"] = bos
    df["choch"] = choch
    return df


def fvg_fast(df):
    """FVG vetorizado — sem loop."""
    df = df.copy()
    h = df["high"].values
    l = df["low"].values
    sz = len(df)
    fvg = np.zeros(sz)
    top = np.full(sz, np.nan)
    bot = np.full(sz, np.nan)

    mask_b = np.zeros(sz, dtype=bool)
    mask_b[2:] = l[2:] > h[:-2]
    idx_b = np.where(mask_b)[0]
    fvg[idx_b] = 1
    top[idx_b] = l[idx_b]
    bot[idx_b] = h[idx_b - 2]

    mask_r = np.zeros(sz, dtype=bool)
    mask_r[2:] = h[2:] < l[:-2]
    idx_r = np.where(mask_r)[0]
    fvg[idx_r] = -1
    top[idx_r] = l[idx_r - 2]
    bot[idx_r] = h[idx_r]

    df["fvg"] = fvg
    df["fvg_top"] = top
    df["fvg_bot"] = bot
    return df


def ob_fast(df, lookback=20):
    """Order Blocks com NumPy."""
    df = df.copy()
    sz = len(df)
    ob = np.zeros(sz)
    ob_t = np.full(sz, np.nan)
    ob_b = np.full(sz, np.nan)
    bos = df["bos"].values
    choch = df["choch"].values
    op = df["open"].values
    cl = df["close"].values
    hi = df["high"].values
    lo = df["low"].values

    for i in range(1, sz):
        sig = int(bos[i]) or int(choch[i])
        if sig == 1:
            for j in range(i - 1, max(0, i - lookback), -1):
                if cl[j] < op[j]:
                    ob[j] = 1
                    ob_t[j] = hi[j]
                    ob_b[j] = lo[j]
                    break
        elif sig == -1:
            for j in range(i - 1, max(0, i - lookback), -1):
                if cl[j] > op[j]:
                    ob[j] = -1
                    ob_t[j] = hi[j]
                    ob_b[j] = lo[j]
                    break

    df["ob"] = ob
    df["ob_top"] = ob_t
    df["ob_bot"] = ob_b
    return df


print("  OK")

# ──────────────────────────────────────────────────────────────
# 3. Benchmark cada indicador
# ──────────────────────────────────────────────────────────────

print("\n[3/6] Benchmark de velocidade (30k candles)...")
print(f"  {'Indicador':<20} {'Tempo':>8}  {'Status'}")
print(f"  {'-' * 45}")

tempos = {}
LIMITE = 5.0  # segundos máximo aceitável


def bench(nome, fn):
    t0 = time.time()
    try:
        resultado = fn()
        elapsed = time.time() - t0
        status = "✓ OK" if elapsed < LIMITE else "⚠ LENTO"
        tempos[nome] = elapsed
        print(f"  {nome:<20} {elapsed:>7.2f}s  {status}")
        return resultado
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {nome:<20} {elapsed:>7.2f}s  ✗ ERRO: {e}")
        tempos[nome] = elapsed
        return None


df_sw = bench("swing_hls", lambda: swing_hls_fast(df_test, 5))
df_bc = bench("bos_choch", lambda: bos_choch_fast(df_sw) if df_sw is not None else None)
df_fvg = bench("fvg", lambda: fvg_fast(df_bc) if df_bc is not None else None)
df_ob = bench("order_blocks", lambda: ob_fast(df_bc, 20) if df_bc is not None else None)
bench("atr", lambda: atr_fast(df_test, 14))
bench("ema", lambda: ema_fast(df_test["close"], 20))
bench("rsi", lambda: rsi_fast(df_test["close"], 14))
bench("macd", lambda: macd_fast(df_test["close"]))
bench("bollinger", lambda: bollinger_fast(df_test["close"]))
bench("adx", lambda: adx_fast(df_test, 14))
bench("supertrend", lambda: supertrend_fast(df_test, 10, 3.0))

# ──────────────────────────────────────────────────────────────
# 4. Teste de backtest completo com versões rápidas
# ──────────────────────────────────────────────────────────────

print("\n[4/6] Testando preparar_indicadores completo (30k candles)...")


def preparar_rapido(df, p):
    df = df.copy()
    sw = p.get("swing_length", 5)
    df = swing_hls_fast(df, sw)
    df = bos_choch_fast(df)
    df = fvg_fast(df)
    df = ob_fast(df, p.get("ob_lookback", 20))
    df["atr"] = atr_fast(df, p.get("atr_period", 14))
    df["atr_s"] = atr_fast(df, p.get("atr_slow_period", 50))
    df["ema_fast"] = ema_fast(df["close"], p.get("ema_fast", 20))
    df["ema_slow"] = ema_fast(df["close"], p.get("ema_slow", 50))
    df["ema_200"] = ema_fast(df["close"], 200)
    df["rsi"] = rsi_fast(df["close"], p.get("rsi_period", 14))
    df["macd"], df["macd_sig"], df["macd_hist"] = macd_fast(
        df["close"],
        p.get("macd_fast", 12),
        p.get("macd_slow", 26),
        p.get("macd_sig", 9),
    )
    df["bb_up"], df["bb_mid"], df["bb_lo"] = bollinger_fast(
        df["close"],
        p.get("bb_period", 20),
        p.get("bb_std", 2.0),
    )
    df["bb_width"] = (df["bb_up"] - df["bb_lo"]) / df["bb_mid"].replace(0, np.nan)
    df["adx"] = adx_fast(df, p.get("adx_period", 14))
    df["supertrend"] = supertrend_fast(df, p.get("st_period", 10), p.get("st_mult", 3.0))
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["roc"] = df["close"].pct_change(p.get("roc_period", 10)) * 100

    h = df.index.hour
    sessao = p.get("gene_sessao", "DIA_INTEIRO")
    if sessao == "MANHA":
        df["na_sessao"] = (h >= 9) & (h < 12)
    elif sessao == "TARDE":
        df["na_sessao"] = (h >= 13) & (h < 17)
    elif sessao == "LONDON_OPEN":
        df["na_sessao"] = (h >= 9) & (h < 11)
    elif sessao == "NY_OPEN":
        df["na_sessao"] = (h >= 11) & (h < 14)
    elif sessao == "FECHAMENTO":
        df["na_sessao"] = (h >= 16) & (h < 18)
    elif sessao == "SEM_ALMOCO":
        df["na_sessao"] = ((h >= 9) & (h < 12)) | ((h >= 13) & (h < 18))
    else:
        df["na_sessao"] = (h >= 9) & (h < 18)

    return df


p_test = {
    "swing_length": 5,
    "ob_lookback": 20,
    "atr_period": 14,
    "atr_slow_period": 50,
    "ema_fast": 20,
    "ema_slow": 50,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "adx_period": 14,
    "st_period": 10,
    "st_mult": 3.0,
    "roc_period": 10,
    "gene_sessao": "DIA_INTEIRO",
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_sig": 9,
}

t0 = time.time()
df_prep = preparar_rapido(df_test.copy(), p_test)
elapsed = time.time() - t0
print(f"  preparar_indicadores: {elapsed:.2f}s {'✓ OK' if elapsed < 10 else '⚠ AINDA LENTO'}")

# ──────────────────────────────────────────────────────────────
# 5. Aplicar correção no arquivo
# ──────────────────────────────────────────────────────────────

print("\n[5/6] Aplicando correções no strategy_composer.py...")

arquivo = "/workspace/strategy_composer/strategy_composer.py"
content = open(arquivo, "r", encoding="utf-8").read()

idx_start = content.find("def preparar_indicadores(")
if idx_start == -1:
    raise RuntimeError("Função preparar_indicadores não encontrada no arquivo.")

idx_end = content.find("\ndef ", idx_start + 10)
if idx_end == -1:
    idx_end = len(content)

nova_func = '''def preparar_indicadores(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Calcula todos os indicadores — versao otimizada com NumPy puro."""
    df = df.copy()

    # --- SMC ---
    sw = p.get("swing_length", 5)
    h_arr = df["high"].values
    l_arr = df["low"].values
    sz = len(df)
    sh_arr = np.zeros(sz)
    sl_arr = np.zeros(sz)

    for i in range(sw, sz - sw):
        wh = h_arr[i-sw:i+sw+1]
        wl = l_arr[i-sw:i+sw+1]
        if h_arr[i] == wh.max() and h_arr[i] > h_arr[i-1] and h_arr[i] > h_arr[i+1]:
            sh_arr[i] = h_arr[i]
        if l_arr[i] == wl.min() and l_arr[i] < l_arr[i-1] and l_arr[i] < l_arr[i+1]:
            sl_arr[i] = l_arr[i]

    df["sh"] = sh_arr
    df["sl"] = sl_arr

    # BOS/CHoCH
    bos_arr = np.zeros(sz, dtype=int)
    choch_arr = np.zeros(sz, dtype=int)
    lsh = lsl = None
    trend = 0

    for i in range(1, sz):
        if sh_arr[i] > 0:
            if lsh is not None and sh_arr[i] > lsh:
                if trend == 1:
                    bos_arr[i] = 1
                else:
                    choch_arr[i] = 1
                    trend = 1
            lsh = sh_arr[i]

        if sl_arr[i] > 0:
            if lsl is not None and sl_arr[i] < lsl:
                if trend == -1:
                    bos_arr[i] = -1
                else:
                    choch_arr[i] = -1
                    trend = -1
            lsl = sl_arr[i]

    df["bos"] = bos_arr
    df["choch"] = choch_arr

    # FVG vetorizado
    fvg_arr = np.zeros(sz)
    fvg_top = np.full(sz, np.nan)
    fvg_bot = np.full(sz, np.nan)

    mask_b = np.zeros(sz, dtype=bool)
    mask_r = np.zeros(sz, dtype=bool)
    mask_b[2:] = l_arr[2:] > h_arr[:-2]
    mask_r[2:] = h_arr[2:] < l_arr[:-2]

    bi = np.where(mask_b)[0]
    ri = np.where(mask_r)[0]

    fvg_arr[bi] = 1
    fvg_top[bi] = l_arr[bi]
    fvg_bot[bi] = h_arr[bi - 2]

    fvg_arr[ri] = -1
    fvg_top[ri] = l_arr[ri - 2]
    fvg_bot[ri] = h_arr[ri]

    df["fvg"] = fvg_arr
    df["fvg_top"] = fvg_top
    df["fvg_bot"] = fvg_bot

    # Order Blocks
    lb = p.get("ob_lookback", 20)
    ob_arr = np.zeros(sz)
    ob_top = np.full(sz, np.nan)
    ob_bot = np.full(sz, np.nan)
    op_arr = df["open"].values
    cl_arr = df["close"].values

    for i in range(1, sz):
        sig = int(bos_arr[i]) or int(choch_arr[i])
        if sig == 1:
            for j in range(i-1, max(0, i-lb), -1):
                if cl_arr[j] < op_arr[j]:
                    ob_arr[j] = 1
                    ob_top[j] = h_arr[j]
                    ob_bot[j] = l_arr[j]
                    break
        elif sig == -1:
            for j in range(i-1, max(0, i-lb), -1):
                if cl_arr[j] > op_arr[j]:
                    ob_arr[j] = -1
                    ob_top[j] = h_arr[j]
                    ob_bot[j] = l_arr[j]
                    break

    df["ob"] = ob_arr
    df["ob_top"] = ob_top
    df["ob_bot"] = ob_bot

    # --- Indicadores vetorizados ---
    close = df["close"]
    h_s = df["high"]
    l_s = df["low"]
    c_s1 = close.shift(1)

    tr = pd.concat([h_s - l_s, (h_s - c_s1).abs(), (l_s - c_s1).abs()], axis=1).max(axis=1)
    ap = p.get("atr_period", 14)
    asp = p.get("atr_slow_period", 50)
    df["atr"] = tr.rolling(ap).mean()
    df["atr_s"] = tr.rolling(asp).mean()

    df["ema_fast"] = close.ewm(span=p.get("ema_fast", 20), adjust=False).mean()
    df["ema_slow"] = close.ewm(span=p.get("ema_slow", 50), adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()

    d = close.diff()
    g = d.clip(lower=0).rolling(p.get("rsi_period", 14)).mean()
    ls = (-d.clip(upper=0)).rolling(p.get("rsi_period", 14)).mean()
    df["rsi"] = 100 - (100 / (1 + g / ls.replace(0, np.nan)))

    ef = close.ewm(span=p.get("macd_fast", 12), adjust=False).mean()
    es = close.ewm(span=p.get("macd_slow", 26), adjust=False).mean()
    mac = ef - es
    sig = mac.ewm(span=p.get("macd_sig", 9), adjust=False).mean()
    df["macd"] = mac
    df["macd_sig"] = sig
    df["macd_hist"] = mac - sig

    bp = p.get("bb_period", 20)
    bs = p.get("bb_std", 2.0)
    ma = close.rolling(bp).mean()
    dev = close.rolling(bp).std()
    df["bb_up"] = ma + bs * dev
    df["bb_mid"] = ma
    df["bb_lo"] = ma - bs * dev
    df["bb_width"] = (df["bb_up"] - df["bb_lo"]) / ma.replace(0, np.nan)

    adp = p.get("adx_period", 14)
    up_m = np.diff(h_arr, prepend=h_arr[0])
    dn_m = -np.diff(l_arr, prepend=l_arr[0])
    pdm = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=df.index)
    atr_adx = df["atr"].values
    pdi = 100 * pdm.rolling(adp).mean() / (atr_adx + 1e-9)
    mdi = 100 * mdm.rolling(adp).mean() / (atr_adx + 1e-9)
    denom = (pdi + mdi).replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / denom
    df["adx"] = dx.rolling(adp).mean()

    # Supertrend
    atr_v = df["atr"].values
    hl2 = (h_arr + l_arr) / 2
    stm = p.get("st_mult", 3.0)
    up_st = hl2 - stm * atr_v
    dn_st = hl2 + stm * atr_v
    dir_st = np.ones(sz)

    for i in range(1, sz):
        if np.isnan(atr_v[i]):
            dir_st[i] = dir_st[i-1]
            continue
        up_st[i] = max(up_st[i], up_st[i-1]) if cl_arr[i-1] > up_st[i-1] else up_st[i]
        dn_st[i] = min(dn_st[i], dn_st[i-1]) if cl_arr[i-1] < dn_st[i-1] else dn_st[i]
        if cl_arr[i] > dn_st[i-1]:
            dir_st[i] = 1
        elif cl_arr[i] < up_st[i-1]:
            dir_st[i] = -1
        else:
            dir_st[i] = dir_st[i-1]

    df["supertrend"] = dir_st

    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["roc"] = close.pct_change(p.get("roc_period", 10)) * 100

    hour = df.index.hour
    sessao = p.get("gene_sessao", "DIA_INTEIRO")
    if sessao == "MANHA":
        df["na_sessao"] = (hour >= 9) & (hour < 12)
    elif sessao == "TARDE":
        df["na_sessao"] = (hour >= 13) & (hour < 17)
    elif sessao == "LONDON_OPEN":
        df["na_sessao"] = (hour >= 9) & (hour < 11)
    elif sessao == "NY_OPEN":
        df["na_sessao"] = (hour >= 11) & (hour < 14)
    elif sessao == "FECHAMENTO":
        df["na_sessao"] = (hour >= 16) & (hour < 18)
    elif sessao == "SEM_ALMOCO":
        df["na_sessao"] = ((hour >= 9) & (hour < 12)) | ((hour >= 13) & (hour < 18))
    else:
        df["na_sessao"] = (hour >= 9) & (hour < 18)

    return df
'''

content = content[:idx_start] + nova_func + content[idx_end:]
with open(arquivo, "w", encoding="utf-8") as f:
    f.write(content)

print("  OK - preparar_indicadores substituido!")

# ──────────────────────────────────────────────────────────────
# 6. Validar correção
# ──────────────────────────────────────────────────────────────

print("\n[6/6] Validando correcao final...")

if "strategy_composer" in sys.modules:
    del sys.modules["strategy_composer"]

import strategy_composer as sc

sc.MIN_TRADES = 10
sc.MIN_PF = 0.3
sc.MAX_DD = -50.0

df_val = sc.carregar_csv("/workspace/wdo_2025.csv")
df_val = df_val.iloc[:30000]

genes = ["BREAKOUT_VOL", "CHoCH_FVG", "MACD_SIGNAL", "MOMENTUM_BREAK"]
print(f"\n  {'Gene':<20} {'Tempo':>7}  {'Trades':>7}  {'PF':>6}  {'Status'}")
print(f"  {'-' * 55}")

todos_ok = True
for gene in genes:
    p = sc.gerar_params_aleatorio()
    p.update(
        {
            "gene_entrada": gene,
            "gene_filtro_t": "NENHUM",
            "gene_filtro_v": "NENHUM",
            "gene_sessao": "DIA_INTEIRO",
            "gene_saida": "RR_FIXO",
            "rr_min": 1.5,
        }
    )
    t0 = time.time()
    m = sc.rodar_backtest(df_val.copy(), p)
    el = time.time() - t0
    tr = m.get("total_trades", 0) if m else 0
    pf = m.get("profit_factor", 0) if m else 0
    ok = el < 15 and tr >= 5
    if not ok:
        todos_ok = False
    status = "✓" if ok else "✗"
    print(f"  {gene:<20} {el:>6.1f}s  {tr:>7}  {pf:>6.3f}  {status}")

print()
if todos_ok:
    print("  ✅ TUDO OK! Rode agora:")
    print("     python3.13 strategy_composer.py --mini")
else:
    print("  ⚠ Ainda há problemas. Verifique os itens marcados com ✗")

print("\n" + "=" * 60)