"""
╔══════════════════════════════════════════════════════════════════════╗
║   STRATEGY COMPOSER — WDO (B3)                                      ║
║   Descobre automaticamente as melhores combinações de:              ║
║   - Sinais de entrada (CHoCH, Breakout, RSI, EMA cross…)           ║
║   - Filtros de tendência (EMA, ADX, sem filtro…)                   ║
║   - Filtros de volatilidade (ATR, Bollinger, sem filtro…)          ║
║   - Filtros de sessão (manhã, tarde, dia inteiro…)                 ║
║   - Gestão de saída (RR fixo, Trailing, TP parcial…)               ║
║   via Algoritmo Genético + joblib/loky (32 cores)                  ║
╠══════════════════════════════════════════════════════════════════════╣
║   INSTALAÇÃO:                                                       ║
║   pip install pandas numpy joblib optuna quantstats tqdm           ║
║                                                                      ║
║   RODAR:                                                            ║
║   python strategy_composer.py --mini   # teste rápido (~1 min)     ║
║   python strategy_composer.py          # completo (horas)          ║
║                                                                      ║
║   IMPORTANTE: Coloque o CSV em /workspace/wdo_2025.csv             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import random
import logging
import warnings
import itertools
from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("COMPOSER")

# joblib/loky — robusto no RunPod Linux, sem deadlock
from joblib import Parallel, delayed

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    log.warning("optuna nao instalado: pip install optuna")

try:
    import quantstats as qs

    QS_OK = True
except ImportError:
    QS_OK = False

# ──────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────────────

CSV_PATH = "/workspace/wdo_2025.csv"
OUT_DIR = "/workspace/composer_output"
N_CORES = min(32, os.cpu_count() or 4)
CAPITAL = 50_000.0
MULT_WDO = 10.0
COMISSAO = 5.0
RISCO_PCT = 0.01
CONTRATOS = 1

# Filtros mínimos de qualidade
MIN_TRADES = 20
MIN_PF = 0.7
MAX_DD = -30.0

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — GENES (blocos modulares da estratégia)
# ══════════════════════════════════════════════════════════════════════

"""
Cada estratégia é composta por 5 genes:

GENE_ENTRADA    → Como detectar sinal de entrada
GENE_FILTRO_T   → Filtro de tendência (ou nenhum)
GENE_FILTRO_V   → Filtro de volatilidade (ou nenhum)
GENE_SESSAO     → Filtro de horário
GENE_SAIDA      → Como sair do trade

Cada gene tem múltiplas opções e parâmetros próprios.
O Algoritmo Genético combina e muta esses genes.
"""

GENE_ENTRADA_OPCOES = [
    "CHoCH_FVG",
    "CHoCH_OB",
    "CHoCH_FVG_OB",
    "LIQ_SWEEP",
    "BREAKOUT_VOL",
    "RSI_EXTREME",
    "EMA_CROSS",
    "BB_REVERSAL",
    "MACD_SIGNAL",
    "DOJI_REVERSAL",
    "ENGULF_SMC",
    "MOMENTUM_BREAK",
]

GENE_FILTRO_T_OPCOES = [
    "NENHUM",
    "EMA_FAST_SLOW",
    "EMA_200",
    "ADX_TREND",
    "HH_HL",
    "MACD_HIST",
    "SUPERTREND",
]

GENE_FILTRO_V_OPCOES = [
    "NENHUM",
    "ATR_EXPANDING",
    "ATR_CONTRACTING",
    "BB_SQUEEZE",
    "ATR_RANGE",
    "VOLUME_ABOVE_MA",
]

GENE_SESSAO_OPCOES = [
    "DIA_INTEIRO",
    "MANHA",
    "TARDE",
    "LONDON_OPEN",
    "NY_OPEN",
    "FECHAMENTO",
    "SEM_ALMOCO",
]

GENE_SAIDA_OPCOES = [
    "RR_FIXO",
    "TRAILING_ATR",
    "TP_PARCIAL",
    "OPOSTO_SIGNAL",
    "TIME_EXIT",
    "BB_OPOSTO",
]

# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — INDICADORES
# ══════════════════════════════════════════════════════════════════════


def calc_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def calc_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def calc_rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(s: pd.Series, fast=12, slow=26, sig=9):
    ema_f = calc_ema(s, fast)
    ema_s = calc_ema(s, slow)
    macd = ema_f - ema_s
    signal = calc_ema(macd, sig)
    hist = macd - signal
    return macd, signal, hist


def calc_adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h - h.shift(1)
    down = l.shift(1) - l
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_n = tr.rolling(n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(n).mean() / atr_n
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(n).mean() / atr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(n).mean()


def calc_bollinger(s: pd.Series, n: int = 20, std: float = 2.0):
    ma = s.rolling(n).mean()
    dev = s.rolling(n).std()
    return ma + std * dev, ma, ma - std * dev


def calc_supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    atr = calc_atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    dir_ = pd.Series(np.ones(len(df)), index=df.index)
    for i in range(1, len(df)):
        up.iloc[i] = max(up.iloc[i], up.iloc[i - 1]) if df["close"].iloc[i - 1] > up.iloc[i - 1] else up.iloc[i]
        dn.iloc[i] = min(dn.iloc[i], dn.iloc[i - 1]) if df["close"].iloc[i - 1] < dn.iloc[i - 1] else dn.iloc[i]
        if df["close"].iloc[i] > dn.iloc[i - 1]:
            dir_.iloc[i] = 1
        elif df["close"].iloc[i] < up.iloc[i - 1]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]
    return dir_


def calc_swing_hls(df: pd.DataFrame, n: int = 5):
    h, l = df["high"].values, df["low"].values
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


def calc_bos_choch(df: pd.DataFrame):
    df = df.copy()
    df["bos"] = 0
    df["choch"] = 0
    lsh = lsl = None
    trend = 0
    for i in range(1, len(df)):
        sv = df.iloc[i]["sh"]
        lv = df.iloc[i]["sl"]
        if sv > 0:
            if lsh is not None and sv > lsh:
                col = "bos" if trend == 1 else "choch"
                df.iat[i, df.columns.get_loc(col)] = 1
                if trend != 1:
                    trend = 1
            lsh = sv
        if lv > 0:
            if lsl is not None and lv < lsl:
                col = "bos" if trend == -1 else "choch"
                df.iat[i, df.columns.get_loc(col)] = -1
                if trend != -1:
                    trend = -1
            lsl = lv
    return df


def calc_fvg(df: pd.DataFrame):
    df = df.copy()
    df["fvg"] = 0
    df["fvg_top"] = np.nan
    df["fvg_bot"] = np.nan
    h, l = df["high"].values, df["low"].values
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            df.iat[i, df.columns.get_loc("fvg")] = 1
            df.iat[i, df.columns.get_loc("fvg_top")] = l[i]
            df.iat[i, df.columns.get_loc("fvg_bot")] = h[i - 2]
        elif h[i] < l[i - 2]:
            df.iat[i, df.columns.get_loc("fvg")] = -1
            df.iat[i, df.columns.get_loc("fvg_top")] = l[i - 2]
            df.iat[i, df.columns.get_loc("fvg_bot")] = h[i]
    return df


def calc_ob(df: pd.DataFrame, lookback: int = 20):
    df = df.copy()
    df["ob"] = 0
    df["ob_top"] = np.nan
    df["ob_bot"] = np.nan
    for i in range(1, len(df)):
        sig = df.iloc[i]["bos"] or df.iloc[i]["choch"]
        if sig == 1:
            for j in range(i - 1, max(0, i - lookback), -1):
                if df.iloc[j]["close"] < df.iloc[j]["open"]:
                    df.iat[j, df.columns.get_loc("ob")] = 1
                    df.iat[j, df.columns.get_loc("ob_top")] = df.iloc[j]["high"]
                    df.iat[j, df.columns.get_loc("ob_bot")] = df.iloc[j]["low"]
                    break
        elif sig == -1:
            for j in range(i - 1, max(0, i - lookback), -1):
                if df.iloc[j]["close"] > df.iloc[j]["open"]:
                    df.iat[j, df.columns.get_loc("ob")] = -1
                    df.iat[j, df.columns.get_loc("ob_top")] = df.iloc[j]["high"]
                    df.iat[j, df.columns.get_loc("ob_bot")] = df.iloc[j]["low"]
                    break
    return df


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — PREPARAR DADOS COM TODOS OS INDICADORES
# ══════════════════════════════════════════════════════════════════════

def preparar_indicadores(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Calcula todos os indicadores necessários para todos os genes."""
    df = df.copy()

    sw = p.get("swing_length", 5)
    df = calc_swing_hls(df, sw)
    df = calc_bos_choch(df)
    df = calc_fvg(df)
    df = calc_ob(df, p.get("ob_lookback", 20))

    df["atr"] = calc_atr(df, p.get("atr_period", 14))
    df["atr_s"] = calc_atr(df, p.get("atr_slow_period", 50))

    df["ema_fast"] = calc_ema(df["close"], p.get("ema_fast", 20))
    df["ema_slow"] = calc_ema(df["close"], p.get("ema_slow", 50))
    df["ema_200"] = calc_ema(df["close"], 200)

    df["rsi"] = calc_rsi(df["close"], p.get("rsi_period", 14))

    df["macd"], df["macd_sig"], df["macd_hist"] = calc_macd(
        df["close"], p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_sig", 9)
    )

    df["bb_up"], df["bb_mid"], df["bb_lo"] = calc_bollinger(
        df["close"], p.get("bb_period", 20), p.get("bb_std", 2.0)
    )
    bb_w = (df["bb_up"] - df["bb_lo"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_width"] = bb_w

    df["adx"] = calc_adx(df, p.get("adx_period", 14))

    try:
        df["supertrend"] = calc_supertrend(df, p.get("st_period", 10), p.get("st_mult", 3.0))
    except Exception:
        df["supertrend"] = 0

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


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MOTOR DE SINAL
# ══════════════════════════════════════════════════════════════════════

class GeneMotor:
    """Interpreta os genes e gera sinais de trading."""

    def __init__(self, p: dict):
        self.p = p
        self.estado = {
            "ult_choch_bull": -9999,
            "ult_choch_bear": -9999,
            "fvgs_bull": [],
            "fvgs_bear": [],
            "obs_bull": [],
            "obs_bear": [],
            "prev_ema_fast": None,
            "prev_ema_slow": None,
            "prev_macd": None,
            "prev_macd_sig": None,
        }

    def atualizar_estado_smc(self, df: pd.DataFrame, i: int):
        row = df.iloc[i]
        pj = self.p.get("poi_janela", 60)

        if row["choch"] == 1:
            self.estado["ult_choch_bull"] = i
            self.estado["fvgs_bull"] = []
            self.estado["obs_bull"] = []
        if row["choch"] == -1:
            self.estado["ult_choch_bear"] = i
            self.estado["fvgs_bear"] = []
            self.estado["obs_bear"] = []

        if row["fvg"] == 1 and not np.isnan(row.get("fvg_top", np.nan)):
            self.estado["fvgs_bull"].append({"top": row["fvg_top"], "bot": row["fvg_bot"], "i": i})
        if row["fvg"] == -1 and not np.isnan(row.get("fvg_top", np.nan)):
            self.estado["fvgs_bear"].append({"top": row["fvg_top"], "bot": row["fvg_bot"], "i": i})
        if row["ob"] == 1:
            self.estado["obs_bull"].append({"top": row["ob_top"], "bot": row["ob_bot"], "i": i})
        if row["ob"] == -1:
            self.estado["obs_bear"].append({"top": row["ob_top"], "bot": row["ob_bot"], "i": i})

        for k in ["fvgs_bull", "fvgs_bear", "obs_bull", "obs_bear"]:
            self.estado[k] = [x for x in self.estado[k] if i - x["i"] <= pj]

    def sinal_entrada(self, df: pd.DataFrame, i: int) -> Tuple[int, Optional[dict], str]:
        row = df.iloc[i]
        close = row["close"]
        gene = self.p.get("gene_entrada", "CHoCH_FVG")
        cj = self.p.get("choch_janela", 60)

        self.atualizar_estado_smc(df, i)

        if gene in ("CHoCH_FVG", "CHoCH_FVG_OB"):
            if (i - self.estado["ult_choch_bull"]) <= cj:
                for fg in reversed(self.estado["fvgs_bull"]):
                    if fg["bot"] <= close <= fg["top"]:
                        return 1, fg, "CHoCH+FVG"
                if gene == "CHoCH_FVG_OB":
                    for ob in reversed(self.estado["obs_bull"]):
                        if ob["bot"] <= close <= ob["top"]:
                            return 1, ob, "CHoCH+OB"

            if (i - self.estado["ult_choch_bear"]) <= cj:
                for fg in reversed(self.estado["fvgs_bear"]):
                    if fg["bot"] <= close <= fg["top"]:
                        return -1, fg, "CHoCH+FVG"
                if gene == "CHoCH_FVG_OB":
                    for ob in reversed(self.estado["obs_bear"]):
                        if ob["bot"] <= close <= ob["top"]:
                            return -1, ob, "CHoCH+OB"

        elif gene == "CHoCH_OB":
            if (i - self.estado["ult_choch_bull"]) <= cj:
                for ob in reversed(self.estado["obs_bull"]):
                    if ob["bot"] <= close <= ob["top"]:
                        return 1, ob, "CHoCH+OB"
            if (i - self.estado["ult_choch_bear"]) <= cj:
                for ob in reversed(self.estado["obs_bear"]):
                    if ob["bot"] <= close <= ob["top"]:
                        return -1, ob, "CHoCH+OB"

        elif gene == "LIQ_SWEEP":
            if i >= 20:
                rec_h = df["high"].iloc[i - 20:i].max()
                rec_l = df["low"].iloc[i - 20:i].min()
                if row["high"] > rec_h and row["close"] < rec_h:
                    poi = {"top": rec_h + row["atr"] * 0.3, "bot": rec_h - row["atr"] * 0.3}
                    return -1, poi, "LIQ_SWEEP_BEAR"
                if row["low"] < rec_l and row["close"] > rec_l:
                    poi = {"top": rec_l + row["atr"] * 0.3, "bot": rec_l - row["atr"] * 0.3}
                    return 1, poi, "LIQ_SWEEP_BULL"

        elif gene == "BREAKOUT_VOL":
            n = self.p.get("breakout_period", 20)
            if i < n:
                return 0, None, ""
            rh = df["high"].iloc[i - n:i].max()
            rl = df["low"].iloc[i - n:i].min()
            vol_ok = row["volume"] > row["vol_ma"] * self.p.get("vol_mult", 1.3)
            if close > rh and vol_ok:
                poi = {"top": close + row["atr"] * 0.3, "bot": rh - row["atr"] * 0.3}
                return 1, poi, "BREAKOUT_BULL"
            if close < rl and vol_ok:
                poi = {"top": rl + row["atr"] * 0.3, "bot": close - row["atr"] * 0.3}
                return -1, poi, "BREAKOUT_BEAR"

        elif gene == "RSI_EXTREME":
            rlo = self.p.get("rsi_low", 30)
            rhi = self.p.get("rsi_high", 70)
            if np.isnan(row.get("rsi", np.nan)):
                return 0, None, ""
            poi = {"top": close + row["atr"] * 0.5, "bot": close - row["atr"] * 0.5}
            if row["rsi"] < rlo and close < row.get("bb_lo", float("inf")):
                return 1, poi, "RSI_OVERSOLD"
            if row["rsi"] > rhi and close > row.get("bb_up", float("-inf")):
                return -1, poi, "RSI_OVERBOUGHT"

        elif gene == "EMA_CROSS":
            ef = row.get("ema_fast", np.nan)
            es = row.get("ema_slow", np.nan)
            pef = self.estado.get("prev_ema_fast")
            pes = self.estado.get("prev_ema_slow")
            self.estado["prev_ema_fast"] = ef
            self.estado["prev_ema_slow"] = es
            if pef is None or np.isnan(ef) or np.isnan(es):
                return 0, None, ""
            poi = {"top": close + row["atr"] * 0.5, "bot": close - row["atr"] * 0.5}
            if pef <= pes and ef > es:
                return 1, poi, "EMA_CROSS_BULL"
            if pef >= pes and ef < es:
                return -1, poi, "EMA_CROSS_BEAR"

        elif gene == "BB_REVERSAL":
            if np.isnan(row.get("bb_up", np.nan)):
                return 0, None, ""
            poi = {"top": row["bb_mid"] + row["atr"] * 0.3, "bot": row["bb_mid"] - row["atr"] * 0.3}
            if close < row["bb_lo"] and row["rsi"] < 40:
                return 1, poi, "BB_REVERSAL_BULL"
            if close > row["bb_up"] and row["rsi"] > 60:
                return -1, poi, "BB_REVERSAL_BEAR"

        elif gene == "MACD_SIGNAL":
            m = row.get("macd", np.nan)
            ms = row.get("macd_sig", np.nan)
            pm = self.estado.get("prev_macd")
            pms = self.estado.get("prev_macd_sig")
            self.estado["prev_macd"] = m
            self.estado["prev_macd_sig"] = ms
            if pm is None or np.isnan(m) or np.isnan(ms):
                return 0, None, ""
            poi = {"top": close + row["atr"] * 0.5, "bot": close - row["atr"] * 0.5}
            if pm <= pms and m > ms:
                return 1, poi, "MACD_BULL"
            if pm >= pms and m < ms:
                return -1, poi, "MACD_BEAR"

        elif gene == "DOJI_REVERSAL":
            body = abs(row["close"] - row["open"])
            rng = row["high"] - row["low"]
            is_doji = rng > 0 and (body / rng) < 0.25
            if not is_doji:
                return 0, None, ""
            if (i - self.estado["ult_choch_bull"]) <= cj:
                for fg in reversed(self.estado["fvgs_bull"]):
                    if fg["bot"] <= close <= fg["top"]:
                        return 1, fg, "DOJI+FVG"
            if (i - self.estado["ult_choch_bear"]) <= cj:
                for fg in reversed(self.estado["fvgs_bear"]):
                    if fg["bot"] <= close <= fg["top"]:
                        return -1, fg, "DOJI+FVG"

        elif gene == "ENGULF_SMC":
            if i == 0:
                return 0, None, ""
            prev = df.iloc[i - 1]
            bull_engulf = (
                row["close"] > row["open"]
                and prev["close"] < prev["open"]
                and row["close"] > prev["open"]
                and row["open"] < prev["close"]
            )
            bear_engulf = (
                row["close"] < row["open"]
                and prev["close"] > prev["open"]
                and row["close"] < prev["open"]
                and row["open"] > prev["close"]
            )
            if bull_engulf and (i - self.estado["ult_choch_bull"]) <= cj:
                for fg in reversed(self.estado["fvgs_bull"] + self.estado["obs_bull"]):
                    if fg["bot"] <= close <= fg["top"]:
                        return 1, fg, "ENGULF+FVG"
            if bear_engulf and (i - self.estado["ult_choch_bear"]) <= cj:
                for fg in reversed(self.estado["fvgs_bear"] + self.estado["obs_bear"]):
                    if fg["bot"] <= close <= fg["top"]:
                        return -1, fg, "ENGULF+FVG"

        elif gene == "MOMENTUM_BREAK":
            roc = row.get("roc", np.nan)
            if np.isnan(roc):
                return 0, None, ""
            thr = self.p.get("roc_threshold", 0.5)
            poi = {"top": close + row["atr"] * 0.5, "bot": close - row["atr"] * 0.5}
            if roc > thr and row["ema_fast"] > row["ema_slow"]:
                return 1, poi, "MOMENTUM_BULL"
            if roc < -thr and row["ema_fast"] < row["ema_slow"]:
                return -1, poi, "MOMENTUM_BEAR"

        return 0, None, ""

    def filtro_tendencia(self, row: pd.Series, direcao: int) -> bool:
        gene = self.p.get("gene_filtro_t", "NENHUM")
        if gene == "NENHUM":
            return True
        if np.isnan(row.get("ema_fast", np.nan)):
            return True

        if gene == "EMA_FAST_SLOW":
            if direcao == 1:
                return row["ema_fast"] > row["ema_slow"]
            if direcao == -1:
                return row["ema_fast"] < row["ema_slow"]

        elif gene == "EMA_200":
            if np.isnan(row.get("ema_200", np.nan)):
                return True
            if direcao == 1:
                return row["close"] > row["ema_200"]
            if direcao == -1:
                return row["close"] < row["ema_200"]

        elif gene == "ADX_TREND":
            adx = row.get("adx", np.nan)
            if np.isnan(adx):
                return True
            return adx > self.p.get("adx_threshold", 25)

        elif gene == "HH_HL":
            if direcao == 1:
                return row.get("bos", 0) >= 0
            if direcao == -1:
                return row.get("bos", 0) <= 0

        elif gene == "MACD_HIST":
            hist = row.get("macd_hist", np.nan)
            if np.isnan(hist):
                return True
            if direcao == 1:
                return hist > 0
            if direcao == -1:
                return hist < 0

        elif gene == "SUPERTREND":
            st = row.get("supertrend", 0)
            if direcao == 1:
                return st == 1
            if direcao == -1:
                return st == -1

        return True

    def filtro_volatilidade(self, row: pd.Series) -> bool:
        gene = self.p.get("gene_filtro_v", "NENHUM")
        if gene == "NENHUM":
            return True

        atr = row.get("atr", np.nan)
        atr_s = row.get("atr_s", np.nan)
        if np.isnan(atr) or np.isnan(atr_s):
            return True

        if gene == "ATR_EXPANDING":
            return atr > atr_s * self.p.get("atr_expand_mult", 1.0)

        elif gene == "ATR_CONTRACTING":
            return atr < atr_s * self.p.get("atr_contract_mult", 0.8)

        elif gene == "BB_SQUEEZE":
            bw = row.get("bb_width", np.nan)
            if np.isnan(bw):
                return True
            return bw < self.p.get("bb_squeeze_thr", 0.03)

        elif gene == "ATR_RANGE":
            lo = self.p.get("atr_range_lo", 2.0)
            hi = self.p.get("atr_range_hi", 15.0)
            return lo <= atr <= hi

        elif gene == "VOLUME_ABOVE_MA":
            vol_ma = row.get("vol_ma", np.nan)
            if np.isnan(vol_ma) or vol_ma == 0:
                return True
            return row.get("volume", 0) > vol_ma * self.p.get("vol_ma_mult", 1.1)

        return True


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — BACKTEST ENGINE DO COMPOSER
# ══════════════════════════════════════════════════════════════════════

def rodar_backtest(df: pd.DataFrame, p: dict) -> dict:
    """
    Roda o backtest completo de uma combinação de genes.
    Retorna dicionário de métricas ou {} se inválido.
    """
    try:
        df = preparar_indicadores(df, p)
    except Exception:
        return {}

    motor = GeneMotor(p)
    capital = CAPITAL
    equity = [capital]
    trades = []
    em_pos = False
    trade = None

    rr = p.get("rr_min", 2.0)
    am = p.get("atr_mult", 0.5)
    gene_s = p.get("gene_saida", "RR_FIXO")
    max_bars = p.get("time_exit_bars", 20)
    bars_open = 0

    n = len(df)
    for i in range(50, n):
        row = df.iloc[i]

        if em_pos and trade:
            d = trade["d"]
            sl = trade["sl"]
            tp = trade["tp"]
            en = trade["entry"]
            bars_open += 1

            if gene_s == "TRAILING_ATR":
                atr_v = row["atr"] if not np.isnan(row.get("atr", np.nan)) else 5.0
                trail_m = p.get("trail_atr_mult", 1.5)
                if d == 1:
                    new_sl = row["close"] - atr_v * trail_m
                    trade["sl"] = max(trade["sl"], new_sl)
                    sl = trade["sl"]
                else:
                    new_sl = row["close"] + atr_v * trail_m
                    trade["sl"] = min(trade["sl"], new_sl)
                    sl = trade["sl"]

            if gene_s == "TIME_EXIT" and bars_open >= max_bars:
                saida = row["close"]
                pts = (saida - en) * d
                brl = pts * MULT_WDO * CONTRATOS - COMISSAO
                capital += brl
                equity.append(round(capital, 2))
                trade.update(
                    {
                        "saida": saida,
                        "pnl_pts": round(pts, 2),
                        "pnl_brl": round(brl, 2),
                        "resultado": "WIN" if brl > 0 else "LOSS",
                    }
                )
                trades.append(trade)
                em_pos = False
                trade = None
                bars_open = 0
                continue

            if gene_s == "BB_OPOSTO":
                bb_up = row.get("bb_up", np.nan)
                bb_lo = row.get("bb_lo", np.nan)
                if not np.isnan(bb_up):
                    if d == 1 and row["close"] >= bb_up:
                        tp = row["close"]
                    elif d == -1 and row["close"] <= bb_lo:
                        tp = row["close"]
                    trade["tp"] = tp

            if gene_s == "TP_PARCIAL" and not trade.get("tp1_hit"):
                hit_tp1 = (d == 1 and row["high"] >= tp) or (d == -1 and row["low"] <= tp)
                if hit_tp1:
                    trade["tp1_hit"] = True
                    trade["sl"] = en
                    pts1 = (tp - en) * d
                    capital += pts1 * MULT_WDO * CONTRATOS * 0.5 - COMISSAO * 0.5
                    continue

            hit_sl = (d == 1 and row["low"] <= sl) or (d == -1 and row["high"] >= sl)
            hit_tp = (d == 1 and row["high"] >= tp) or (d == -1 and row["low"] <= tp)

            if gene_s == "OPOSTO_SIGNAL":
                sig_op, _, _ = motor.sinal_entrada(df, i)
                if sig_op == -d:
                    hit_tp = True
                    trade["tp"] = row["close"]
                    tp = row["close"]

            if hit_sl or hit_tp:
                saida = sl if hit_sl else tp
                pts = (saida - en) * d
                brl = pts * MULT_WDO * CONTRATOS - COMISSAO
                capital += brl
                equity.append(round(capital, 2))
                trade.update(
                    {
                        "saida": round(saida, 2),
                        "pnl_pts": round(pts, 2),
                        "pnl_brl": round(brl, 2),
                        "resultado": "WIN" if hit_tp else "LOSS",
                    }
                )
                trades.append(trade)
                em_pos = False
                trade = None
                bars_open = 0
            continue

        if not row.get("na_sessao", True):
            motor.atualizar_estado_smc(df, i)
            continue

        direcao, poi, subtipo = motor.sinal_entrada(df, i)
        if direcao == 0 or poi is None:
            continue

        if not motor.filtro_tendencia(row, direcao):
            continue
        if not motor.filtro_volatilidade(row):
            continue

        atr_v = row["atr"] if not np.isnan(row.get("atr", np.nan)) else 5.0
        slip = 1.0

        if direcao == 1:
            entry = row["close"] + slip
            sl_p = poi["bot"] - atr_v * am
        else:
            entry = row["close"] - slip
            sl_p = poi["top"] + atr_v * am

        risk = abs(entry - sl_p)
        if risk <= 0:
            continue

        tp_p = entry + direcao * risk * rr
        rr_real = abs(tp_p - entry) / risk
        if rr_real < rr * 0.9:
            continue

        risco_brl = risk * MULT_WDO * CONTRATOS
        if risco_brl / capital > RISCO_PCT * 6:
            continue

        em_pos = True
        bars_open = 0
        trade = {
            "entry_dt": str(df.index[i])[:16],
            "d": direcao,
            "entry": round(entry, 2),
            "sl": round(sl_p, 2),
            "tp": round(tp_p, 2),
            "rr": round(rr_real, 2),
            "subtipo": subtipo,
        }

    if em_pos and trade:
        last = df.iloc[-1]["close"]
        pts = (last - trade["entry"]) * trade["d"]
        brl = pts * MULT_WDO * CONTRATOS - COMISSAO
        capital += brl
        trade.update(
            {
                "saida": last,
                "pnl_pts": round(pts, 2),
                "pnl_brl": round(brl, 2),
                "resultado": "ABERTO",
            }
        )
        trades.append(trade)
        equity.append(round(capital, 2))

    fechados = [t for t in trades if t.get("resultado") not in ("ABERTO", None)]
    if len(fechados) < MIN_TRADES:
        return {}

    df_t = pd.DataFrame(fechados)
    wins = df_t[df_t["resultado"] == "WIN"]
    loses = df_t[df_t["resultado"] == "LOSS"]
    n = len(df_t)
    wr = len(wins) / n * 100
    avg_w = wins["pnl_brl"].mean() if len(wins) else 0.0
    avg_l = loses["pnl_brl"].mean() if len(loses) else 0.0
    pnl = df_t["pnl_brl"].sum()
    pf = abs(wins["pnl_brl"].sum() / loses["pnl_brl"].sum()) if loses["pnl_brl"].sum() != 0 else 9999.0

    if pf < MIN_PF:
        return {}

    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    mdd = dd.min()

    if mdd < MAX_DD:
        return {}

    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    neg = rets[rets < 0]
    sortino = rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 else 0.0

    try:
        df_t["saida_dt"] = pd.to_datetime(df_t["entry_dt"], errors="coerce")
        monthly = df_t.groupby(df_t["saida_dt"].dt.to_period("M"))["pnl_brl"].sum()
        consist = (monthly > 0).mean() * 100 if len(monthly) > 0 else 0.0
    except Exception:
        consist = 50.0

    return {
        "total_trades": n,
        "wins": int(len(wins)),
        "losses": int(len(loses)),
        "win_rate": round(wr, 2),
        "profit_factor": round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "avg_win_brl": round(avg_w, 2),
        "avg_loss_brl": round(avg_l, 2),
        "avg_rr": round(df_t["rr"].mean(), 2) if "rr" in df_t else 0.0,
        "total_pnl_brl": round(pnl, 2),
        "retorno_pct": round(pnl / CAPITAL * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "consistencia_pct": round(consist, 1),
        "capital_final": round(CAPITAL + pnl, 2),
        "equity": equity,
        "trades": fechados,
    }


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — SCORE MULTI-OBJETIVO
# ══════════════════════════════════════════════════════════════════════

def calcular_score(m: dict) -> float:
    if not m:
        return -999.0
    pf = min(m.get("profit_factor", 0), 10)
    sharpe = min(max(m.get("sharpe_ratio", 0), 0), 8)
    sortino = min(max(m.get("sortino_ratio", 0), 0), 10)
    wr = m.get("win_rate", 0) / 100
    trades = min(m.get("total_trades", 0), 500)
    consist = m.get("consistencia_pct", 0) / 100
    dd = abs(m.get("max_drawdown_pct", 0))
    ret = min(max(m.get("retorno_pct", -100), -100), 200) / 200

    score = (
        pf / 10 * 0.25
        + sharpe / 8 * 0.20
        + sortino / 10 * 0.15
        + wr * 0.12
        + trades / 500 * 0.10
        + consist * 0.10
        + ret * 0.05
        - (dd / 25) * 0.03
    )
    return round(score, 6)


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — WORKER PARALELO
# ══════════════════════════════════════════════════════════════════════

def _worker_combo(args):
    """Worker isolado para joblib — sem estado compartilhado."""
    df_json, params = args
    try:
        df = pd.read_json(df_json)
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower() for c in df.columns]

        global MIN_TRADES, MIN_PF, MAX_DD
        MIN_TRADES = params.get("_min_trades", MIN_TRADES)
        MIN_PF = params.get("_min_pf", MIN_PF)
        MAX_DD = params.get("_max_dd", MAX_DD)

        m = rodar_backtest(df, params)
        if not m:
            return None
        s = calcular_score(m)
        if s <= 0:
            return None
        return {
            "score": s,
            "gene_entrada": params.get("gene_entrada"),
            "gene_filtro_t": params.get("gene_filtro_t"),
            "gene_filtro_v": params.get("gene_filtro_v"),
            "gene_sessao": params.get("gene_sessao"),
            "gene_saida": params.get("gene_saida"),
            "rr_min": params.get("rr_min"),
            "atr_mult": params.get("atr_mult"),
            "choch_janela": params.get("choch_janela"),
            "poi_janela": params.get("poi_janela"),
            "swing_length": params.get("swing_length"),
            **{k: v for k, v in m.items() if k not in ("equity", "trades")},
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — GERADOR DE COMBINAÇÕES
# ══════════════════════════════════════════════════════════════════════

def gerar_params_aleatorio(
    rng=None,
    min_trades: int = MIN_TRADES,
    min_pf: float = MIN_PF,
    max_dd: float = MAX_DD,
) -> dict:
    """Gera uma combinação aleatória de genes e parâmetros."""
    if rng is None:
        rng = random
    return {
        "_min_trades": min_trades,
        "_min_pf": min_pf,
        "_max_dd": max_dd,
        "gene_entrada": rng.choice(GENE_ENTRADA_OPCOES),
        "gene_filtro_t": rng.choice(GENE_FILTRO_T_OPCOES),
        "gene_filtro_v": rng.choice(GENE_FILTRO_V_OPCOES),
        "gene_sessao": rng.choice(GENE_SESSAO_OPCOES),
        "gene_saida": rng.choice(GENE_SAIDA_OPCOES),
        "rr_min": round(rng.uniform(1.0, 3.5), 1),
        "atr_mult": round(rng.uniform(0.3, 1.5), 2),
        "choch_janela": rng.randint(15, 100),
        "poi_janela": rng.randint(20, 130),
        "swing_length": rng.choice([3, 5, 7, 10]),
        "ob_lookback": rng.choice([15, 20, 30]),
        "atr_period": rng.choice([10, 14, 20]),
        "atr_slow_period": rng.choice([30, 50, 100]),
        "ema_fast": rng.choice([10, 20, 30]),
        "ema_slow": rng.choice([40, 50, 100]),
        "rsi_period": rng.choice([9, 14, 21]),
        "rsi_low": rng.randint(20, 35),
        "rsi_high": rng.randint(65, 80),
        "bb_period": rng.choice([15, 20, 25]),
        "bb_std": round(rng.uniform(1.5, 2.5), 1),
        "adx_period": rng.choice([10, 14, 20]),
        "adx_threshold": rng.randint(20, 35),
        "breakout_period": rng.choice([10, 15, 20, 30]),
        "vol_mult": round(rng.uniform(1.0, 2.0), 1),
        "roc_period": rng.choice([5, 10, 15]),
        "roc_threshold": round(rng.uniform(0.3, 1.0), 2),
        "trail_atr_mult": round(rng.uniform(1.0, 2.5), 1),
        "time_exit_bars": rng.randint(10, 50),
        "st_period": rng.choice([7, 10, 14]),
        "st_mult": round(rng.uniform(2.0, 4.0), 1),
        "atr_expand_mult": round(rng.uniform(0.8, 1.3), 2),
        "atr_contract_mult": round(rng.uniform(0.5, 1.0), 2),
        "atr_range_lo": round(rng.uniform(1.0, 3.0), 1),
        "atr_range_hi": round(rng.uniform(10.0, 25.0), 1),
        "vol_ma_mult": round(rng.uniform(1.0, 1.5), 2),
        "macd_fast": rng.choice([8, 12, 16]),
        "macd_slow": rng.choice([20, 26, 34]),
        "macd_sig": rng.choice([7, 9, 12]),
        "bb_squeeze_thr": round(rng.uniform(0.01, 0.05), 3),
    }


def mutar_params(p: dict, taxa: float = 0.3) -> dict:
    """Muta alguns genes/parâmetros mantendo o restante."""
    novo = deepcopy(p)
    if random.random() < taxa:
        novo["gene_entrada"] = random.choice(GENE_ENTRADA_OPCOES)
    if random.random() < taxa:
        novo["gene_filtro_t"] = random.choice(GENE_FILTRO_T_OPCOES)
    if random.random() < taxa:
        novo["gene_filtro_v"] = random.choice(GENE_FILTRO_V_OPCOES)
    if random.random() < taxa:
        novo["gene_sessao"] = random.choice(GENE_SESSAO_OPCOES)
    if random.random() < taxa:
        novo["gene_saida"] = random.choice(GENE_SAIDA_OPCOES)
    for k, v in list(novo.items()):
        if isinstance(v, float) and random.random() < taxa * 0.5:
            novo[k] = round(v * random.uniform(0.7, 1.3), 3)
        elif isinstance(v, int) and random.random() < taxa * 0.5:
            novo[k] = max(1, int(v * random.uniform(0.7, 1.3)))
    return novo


def cruzar_params(p1: dict, p2: dict) -> dict:
    """Combina dois conjuntos de parâmetros (crossover)."""
    filho = {}
    for k in p1:
        filho[k] = p1[k] if random.random() < 0.5 else p2.get(k, p1[k])
    return filho


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — ALGORITMO EVOLUTIVO
# ══════════════════════════════════════════════════════════════════════

def algoritmo_evolutivo(
    df: pd.DataFrame,
    pop_size: int = 100,
    n_gen: int = 30,
    elite_pct: float = 0.2,
    n_cores: int = N_CORES,
) -> List[dict]:
    """
    Algoritmo Genético para descoberta de estratégias.

    A cada geração:
    1. Avalia toda a população em paralelo
    2. Seleciona os melhores
    3. Cria nova geração via crossover + mutação
    4. Mantém Hall of Fame dos melhores
    """
    log.info(f"[EVOL] Iniciando: pop={pop_size} gens={n_gen} cores={n_cores}")

    df_json = df.to_json(date_format="iso")
    hof = []
    random.seed(42)

    populacao = [gerar_params_aleatorio() for _ in range(pop_size)]

    for gen in range(n_gen):
        t0 = time.time()

        args = [(df_json, p) for p in populacao]
        results = Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
            delayed(_worker_combo)(a) for a in args
        )

        validos = [(r, populacao[i]) for i, r in enumerate(results) if r is not None]
        validos.sort(key=lambda x: -x[0]["score"])

        for r, p in validos:
            hof.append({**r, "params": p, "gen": gen})
        hof.sort(key=lambda x: -x["score"])
        hof = hof[:50]

        elapsed = time.time() - t0
        best_score = validos[0][0]["score"] if validos else 0
        best_gene_e = validos[0][1].get("gene_entrada", "?") if validos else "?"
        best_gene_t = validos[0][1].get("gene_filtro_t", "?") if validos else "?"
        best_gene_s = validos[0][1].get("gene_saida", "?") if validos else "?"

        log.info(
            f"  Gen {gen + 1:>3}/{n_gen} | válidos={len(validos):>4} | "
            f"best={best_score:.4f} | "
            f"E={best_gene_e} T={best_gene_t} S={best_gene_s} | "
            f"{elapsed:.0f}s"
        )

        if not validos:
            populacao = [gerar_params_aleatorio() for _ in range(pop_size)]
            continue

        n_elite = max(5, int(len(validos) * elite_pct))
        elite = [p for _, p in validos[:n_elite]]

        nova_pop = list(elite)

        while len(nova_pop) < pop_size * 0.6:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            nova_pop.append(cruzar_params(p1, p2))

        while len(nova_pop) < pop_size * 0.8:
            base = random.choice(elite)
            nova_pop.append(mutar_params(base, taxa=0.3))

        while len(nova_pop) < pop_size:
            nova_pop.append(gerar_params_aleatorio())

        populacao = nova_pop

    return hof


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — RANDOM SEARCH COMPLEMENTAR
# ══════════════════════════════════════════════════════════════════════

def random_search_genes(df: pd.DataFrame, n_trials: int = 500, n_cores: int = N_CORES) -> List[dict]:
    """
    Random Search sobre todas as combinações de genes.
    Garante cobertura ampla do espaço de busca.
    """
    log.info(f"[RANDOM] {n_trials} trials aleatórios | {n_cores} cores")

    combos_base = list(
        itertools.product(
            GENE_ENTRADA_OPCOES[:4],
            GENE_FILTRO_T_OPCOES[:3],
            GENE_FILTRO_V_OPCOES[:3],
            GENE_SESSAO_OPCOES[:3],
            GENE_SAIDA_OPCOES[:3],
        )
    )

    random.seed(123)
    params_list = []

    for ge, gt, gv, gs, gsa in combos_base[: min(n_trials // 3, len(combos_base))]:
        p = gerar_params_aleatorio()
        p.update(
            {
                "gene_entrada": ge,
                "gene_filtro_t": gt,
                "gene_filtro_v": gv,
                "gene_sessao": gs,
                "gene_saida": gsa,
            }
        )
        params_list.append(p)

    while len(params_list) < n_trials:
        params_list.append(gerar_params_aleatorio())

    df_json = df.to_json(date_format="iso")
    args = [(df_json, p) for p in params_list]

    results = Parallel(n_jobs=n_cores, backend="loky", verbose=0)(
        delayed(_worker_combo)(a) for a in args
    )

    validos = []
    for i, r in enumerate(results):
        if r is not None:
            validos.append({**r, "params": params_list[i]})

    validos.sort(key=lambda x: -x["score"])
    log.info(f"[RANDOM] {len(validos)} válidos de {n_trials}")
    return validos


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 11 — ANÁLISE DE COMBINAÇÕES
# ══════════════════════════════════════════════════════════════════════

def analisar_combinacoes(resultados: List[dict]) -> dict:
    """
    Analisa quais genes aparecem mais nos top resultados.
    """
    if not resultados:
        return {}

    top = resultados[: min(50, len(resultados))]

    analise = {
        "gene_entrada": {},
        "gene_filtro_t": {},
        "gene_filtro_v": {},
        "gene_sessao": {},
        "gene_saida": {},
        "combinacoes_top5": [],
    }

    for r in top:
        for gene_key in ["gene_entrada", "gene_filtro_t", "gene_filtro_v", "gene_sessao", "gene_saida"]:
            val = r.get(gene_key, "?")
            if val not in analise[gene_key]:
                analise[gene_key][val] = {"count": 0, "score_sum": 0, "pf_sum": 0}
            analise[gene_key][val]["count"] += 1
            analise[gene_key][val]["score_sum"] += r.get("score", 0)
            analise[gene_key][val]["pf_sum"] += r.get("profit_factor", 0)

    for gene_key in ["gene_entrada", "gene_filtro_t", "gene_filtro_v", "gene_sessao", "gene_saida"]:
        for val in analise[gene_key]:
            d = analise[gene_key][val]
            d["score_avg"] = round(d["score_sum"] / d["count"], 4)
            d["pf_avg"] = round(d["pf_sum"] / d["count"], 3)
        analise[gene_key] = dict(sorted(analise[gene_key].items(), key=lambda x: -x[1]["score_avg"]))

    analise["combinacoes_top5"] = [
        {
            "rank": i + 1,
            "entrada": r.get("gene_entrada"),
            "filtro_t": r.get("gene_filtro_t"),
            "filtro_v": r.get("gene_filtro_v"),
            "sessao": r.get("gene_sessao"),
            "saida": r.get("gene_saida"),
            "score": r.get("score"),
            "pf": r.get("profit_factor"),
            "wr": r.get("win_rate"),
            "trades": r.get("total_trades"),
            "dd": r.get("max_drawdown_pct"),
            "retorno": r.get("retorno_pct"),
        }
        for i, r in enumerate(resultados[:5])
    ]

    return analise


def exibir_analise(analise: dict):
    """Exibe relatório de análise de combinações."""
    if not analise:
        return

    print(f"\n{'═' * 70}")
    print("  ANÁLISE — QUAIS GENES FUNCIONAM MELHOR")
    print(f"{'═' * 70}")

    gene_labels = {
        "gene_entrada": "SINAL DE ENTRADA",
        "gene_filtro_t": "FILTRO DE TENDÊNCIA",
        "gene_filtro_v": "FILTRO DE VOLATILIDADE",
        "gene_sessao": "SESSÃO",
        "gene_saida": "GESTÃO DE SAÍDA",
    }

    for gene_key, label in gene_labels.items():
        dados = analise.get(gene_key, {})
        print(f"\n  {label}:")
        print(f"  {'Gene':<25} {'Aparições':>10} {'Score Médio':>12} {'PF Médio':>10}")
        print(f"  {'─' * 60}")
        for idx, (val, d) in enumerate(list(dados.items())[:6]):
            star = " ★" if idx == 0 else "  "
            print(
                f"{star} {val:<25} {d['count']:>10} "
                f"{d['score_avg']:>12.4f} {d['pf_avg']:>10.3f}"
            )

    print(f"\n{'═' * 70}")
    print("  TOP 5 COMBINAÇÕES COMPLETAS")
    print(f"{'═' * 70}")
    for c in analise.get("combinacoes_top5", []):
        print(
            f"\n  #{c['rank']}: {c['entrada']} + {c['filtro_t']} + "
            f"{c['filtro_v']} + {c['sessao']} + {c['saida']}"
        )
        print(
            f"      Score={c['score']:.4f} PF={c['pf']} "
            f"WR={c['wr']}% Trades={c['trades']} "
            f"DD={c['dd']}% Ret={c['retorno']}%"
        )
    print(f"{'═' * 70}")


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 12 — WALK-FORWARD FINAL
# ══════════════════════════════════════════════════════════════════════

def walk_forward_final(df: pd.DataFrame, params: dict, n_splits: int = 5) -> dict:
    """Walk-Forward para validar a melhor estratégia encontrada."""
    resultados = []
    step = len(df) // n_splits
    log.info(f"[WF] {n_splits} splits para validação final")

    for i in range(n_splits - 1):
        inicio = i * step
        fim = (i + 2) * step
        split = inicio + int((fim - inicio) * 0.7)
        df_tr = df.iloc[inicio:split]
        df_te = df.iloc[split:fim]

        if len(df_tr) < 500 or len(df_te) < 100:
            continue

        m_tr = rodar_backtest(df_tr.copy(), params)
        m_te = rodar_backtest(df_te.copy(), params)
        s_tr = calcular_score(m_tr)
        s_te = calcular_score(m_te)

        log.info(
            f"  Split {i + 1}: TRAIN score={s_tr:.4f} WR={m_tr.get('win_rate', 0) if m_tr else 0}% "
            f"| TEST score={s_te:.4f} WR={m_te.get('win_rate', 0) if m_te else 0}%"
        )

        resultados.append(
            {
                "split": i + 1,
                "score_train": s_tr,
                "score_test": s_te,
                "train": {k: v for k, v in (m_tr or {}).items() if k not in ("equity", "trades")},
                "test": {k: v for k, v in (m_te or {}).items() if k not in ("equity", "trades")},
            }
        )

    scores_test = [r["score_test"] for r in resultados if r["score_test"] > 0]
    lucrativos = [r for r in resultados if r["test"].get("total_pnl_brl", 0) > 0]

    return {
        "splits": resultados,
        "wf_score_medio": round(np.mean(scores_test), 4) if scores_test else -999,
        "splits_lucro": len(lucrativos),
        "total_splits": len(resultados),
    }


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 13 — SALVAR OUTPUTS
# ══════════════════════════════════════════════════════════════════════

def salvar_outputs(hof: List[dict], analise: dict, wf: dict, m_final: dict, bt_equity: list, bt_trades: list):
    os.makedirs(OUT_DIR, exist_ok=True)

    ranking_export = []
    for r in hof:
        row = {k: v for k, v in r.items() if k not in ("equity", "trades", "params")}
        if "params" in r:
            row["params"] = json.dumps(r["params"], ensure_ascii=False)
        ranking_export.append(row)
    pd.DataFrame(ranking_export).to_csv(f"{OUT_DIR}/composer_ranking.csv", index=False)

    with open(f"{OUT_DIR}/composer_analise.json", "w", encoding="utf-8") as f:
        json.dump(analise, f, ensure_ascii=False, indent=2, default=str)

    with open(f"{OUT_DIR}/composer_walkforward.json", "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=2, default=str)

    dashboard = {
        "metricas": m_final,
        "equity_curve": bt_equity,
        "trades": bt_trades,
        "walk_forward": wf.get("splits", []),
        "gerado_em": datetime.now().isoformat(),
        "top_genes": analise.get("combinacoes_top5", []),
    }
    with open(f"{OUT_DIR}/resultado_composer.json", "w", encoding="utf-8") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2, default=str)

    if bt_trades:
        pd.DataFrame(bt_trades).to_csv(f"{OUT_DIR}/composer_trades.csv", index=False)

    if QS_OK and bt_equity and len(bt_equity) > 10:
        try:
            eq = pd.Series(bt_equity)
            ret = eq.pct_change().dropna()
            ret.index = pd.date_range(end=datetime.now(), periods=len(ret), freq="B")
            qs.reports.html(ret, output=f"{OUT_DIR}/composer_quantstats.html", title="Strategy Composer WDO")
            log.info("[OUT] composer_quantstats.html")
        except Exception as e:
            log.warning(f"[OUT] QuantStats: {e}")

    log.info(f"[OUT] Todos os arquivos em {OUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 14 — CARREGAR DADOS
# ══════════════════════════════════════════════════════════════════════

def carregar_csv(path: str) -> pd.DataFrame:
    log.info(f"[DATA] Carregando {path}...")
    try:
        df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime", sep=",")
        df.columns = [c.lower().strip() for c in df.columns]
        needed = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in needed if c in df.columns]]
        df = df[df.index.dayofweek < 5]
        df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        df = df.dropna()
        df = df[df["close"] > 0]
        ret = df["close"].pct_change().abs()
        df = df[ret < 0.05]
        log.info(f"[DATA] {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}")
        return df
    except FileNotFoundError:
        log.warning(f"[DATA] {path} não encontrado. Usando sintético.")
        return _sintetico_df()


def _sintetico_df(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2024-01-02 09:00", periods=n * 3, freq="5min")
    idx = idx[(idx.dayofweek < 5) & (idx.hour >= 9) & (idx.hour < 18)][:n]
    price, op_, hi, lo, cl, vo = 5150.0, [], [], [], [], []
    regime, dur = 1, 0
    for _ in idx:
        dur += 1
        if dur > np.random.randint(60, 300):
            regime = np.random.choice([1, -1, 1, 0], p=[0.4, 0.3, 0.2, 0.1])
            dur = 0
        prev = price
        price = max(4500, min(6500, price * (1 + regime * 0.0003 + np.random.normal(0, 0.0008))))
        sp = abs(np.random.normal(0, 2.5))
        op_.append(prev)
        hi.append(max(prev, price) + sp)
        lo.append(min(prev, price) - sp)
        cl.append(price)
        vo.append(int(np.random.lognormal(6, 0.8)))
    df = pd.DataFrame({"open": op_, "high": hi, "low": lo, "close": cl, "volume": vo}, index=idx)
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    log.info(f"[DATA] Sintético: {len(df):,} candles")
    return df


# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 15 — MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    MINI = "--mini" in sys.argv

    print("╔" + "═" * 68 + "╗")
    print("║  STRATEGY COMPOSER — WDO (B3)                                    ║")
    print("║  Descobre automaticamente as melhores combinações de estratégias ║")
    mode = "MODO MINI (~2min)" if MINI else f"MODO COMPLETO — {N_CORES} CORES"
    print(f"║  {mode:<68}║")
    print("╚" + "═" * 68 + "╝")

    if MINI:
        print("""
Genes testados:
- Entradas  : CHoCH+FVG, CHoCH+OB, Breakout, RSI, EMA Cross…
- Filtros T : EMA, ADX, Supertrend, sem filtro…
- Filtros V : ATR expanding, BB squeeze, volume, sem filtro…
- Sessões   : manhã, tarde, NY open, dia inteiro…
- Saídas    : RR fixo, trailing, TP parcial, sinal oposto…
""")

    t0 = time.time()

    log.info("[1/5] Carregando dados...")
    df = carregar_csv(CSV_PATH)

    split = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    log.info(f"     In-sample: {len(df_ins):,} | Out-sample: {len(df_oos):,}")

    os.makedirs(OUT_DIR, exist_ok=True)
    todos_resultados = []

    if MINI:
        log.info("[MINI] Random Search (30 trials)...")
        res = random_search_genes(df_ins, n_trials=30, n_cores=min(4, N_CORES))
        todos_resultados.extend(res)

        if not todos_resultados:
            log.warning("[MINI] Nenhuma combinação válida. Verifique o CSV.")
            print("\n  ⚠ Sem resultados. Tente com mais dados ou ajuste MIN_TRADES.")
            return

        melhor = todos_resultados[0]
        print("\n  ✅ MINI OK! Melhor combinação encontrada:")
        print(f"  Entrada  : {melhor.get('gene_entrada')}")
        print(f"  Filtro T : {melhor.get('gene_filtro_t')}")
        print(f"  Filtro V : {melhor.get('gene_filtro_v')}")
        print(f"  Sessão   : {melhor.get('gene_sessao')}")
        print(f"  Saída    : {melhor.get('gene_saida')}")
        print(f"  Score    : {melhor.get('score'):.4f}")
        print(f"  PF       : {melhor.get('profit_factor')}")
        print(f"  WR       : {melhor.get('win_rate')}%")
        print(f"  Trades   : {melhor.get('total_trades')}")
        print("\n  Rode sem --mini para o processo completo!\n")
        return

    log.info(f"\n[2/5] Random Search (800 trials | {N_CORES} cores)...")
    res_random = random_search_genes(df_ins, n_trials=800, n_cores=N_CORES)
    todos_resultados.extend(res_random)
    log.info(f"     {len(res_random)} válidos encontrados")

    log.info(f"\n[3/5] Algoritmo Evolutivo ({N_CORES} cores)...")
    hof_evol = algoritmo_evolutivo(
        df_ins,
        pop_size=80,
        n_gen=25,
        n_cores=N_CORES,
    )
    todos_resultados.extend(hof_evol)
    log.info(f"     {len(hof_evol)} elites evolutivos")

    todos_resultados.sort(key=lambda x: -x.get("score", -999))
    log.info(f"\n[4/5] Total válidos: {len(todos_resultados)} | Analisando...")

    analise = analisar_combinacoes(todos_resultados)
    exibir_analise(analise)

    if not todos_resultados:
        log.error("Nenhuma estratégia válida encontrada.")
        return

    melhor = todos_resultados[0]
    params_best = melhor.get("params", {})
    if not params_best:
        params_best = {
            k: melhor[k]
            for k in melhor
            if k.startswith("gene_")
            or k in (
                "rr_min",
                "atr_mult",
                "choch_janela",
                "poi_janela",
                "swing_length",
                "ob_lookback",
                "atr_period",
            )
        }

    log.info("\n  ★ MELHOR ESTRATÉGIA:")
    log.info(f"    Entrada  : {melhor.get('gene_entrada')}")
    log.info(f"    Filtro T : {melhor.get('gene_filtro_t')}")
    log.info(f"    Filtro V : {melhor.get('gene_filtro_v')}")
    log.info(f"    Sessão   : {melhor.get('gene_sessao')}")
    log.info(f"    Saída    : {melhor.get('gene_saida')}")
    log.info(f"    Score    : {melhor.get('score'):.4f}")
    log.info(
        f"    PF       : {melhor.get('profit_factor')} | "
        f"WR={melhor.get('win_rate')}% | "
        f"Trades={melhor.get('total_trades')}"
    )

    log.info("\n[4/5] Backtest completo dataset...")
    m_final = rodar_backtest(df.copy(), params_best)

    log.info("[4/5] Backtest Out-of-Sample...")
    m_oos = rodar_backtest(df_oos.copy(), params_best)
    if m_oos:
        log.info(
            f"  OOS: PF={m_oos['profit_factor']} WR={m_oos['win_rate']}% "
            f"Trades={m_oos['total_trades']} PnL=R${m_oos['total_pnl_brl']:,.0f}"
        )

    log.info("\n[5/5] Walk-Forward Validation...")
    wf = walk_forward_final(df, params_best, n_splits=5)
    log.info(
        f"  WF Score médio OOS: {wf['wf_score_medio']:.4f} | "
        f"Splits lucrativos: {wf['splits_lucro']}/{wf['total_splits']}"
    )

    log.info("\n[OUT] Salvando resultados...")
    bt_equity = m_final.get("equity", []) if m_final else []
    bt_trades = m_final.get("trades", []) if m_final else []
    m_export = {k: v for k, v in (m_final or {}).items() if k not in ("equity", "trades")}
    salvar_outputs(todos_resultados[:50], analise, wf, m_export, bt_equity, bt_trades)

    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print("  STRATEGY COMPOSER — RESULTADO FINAL")
    print(f"{'═' * 70}")
    print("  Estratégia vencedora:")
    print(f"    Entrada  : {melhor.get('gene_entrada')}")
    print(f"    Filtro T : {melhor.get('gene_filtro_t')}")
    print(f"    Filtro V : {melhor.get('gene_filtro_v')}")
    print(f"    Sessão   : {melhor.get('gene_sessao')}")
    print(f"    Saída    : {melhor.get('gene_saida')}")
    if m_final:
        print(f"  Profit Factor    : {m_final.get('profit_factor', 0)}")
        print(f"  Win Rate         : {m_final.get('win_rate', 0)}%")
        print(f"  Sharpe Ratio     : {m_final.get('sharpe_ratio', 0)}")
        print(f"  Max Drawdown     : {m_final.get('max_drawdown_pct', 0)}%")
        print(f"  Total Trades     : {m_final.get('total_trades', 0)}")
        print(f"  Retorno Total    : {m_final.get('retorno_pct', 0)}%")
        print(f"  Capital Final    : R$ {m_final.get('capital_final', 0):,.2f}")
    print(f"  WF Score OOS     : {wf['wf_score_medio']}")
    print(f"  Total válidos    : {len(todos_resultados)}")
    print(f"  Tempo total      : {elapsed / 60:.1f} minutos")
    print(f"  Outputs em       : {OUT_DIR}/")
    print(f"{'═' * 70}")
    print("\n  Arquivos gerados:")
    print("    composer_ranking.csv      — ranking de todas as combinações")
    print("    composer_analise.json     — quais genes funcionam melhor")
    print("    resultado_composer.json   — carregue no smc_dashboard.html")
    print("    composer_trades.csv       — lista de todos os trades")
    print("    composer_quantstats.html  — relatório detalhado")


if __name__ == "__main__":
    main()