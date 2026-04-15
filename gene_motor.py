# ══════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — GENE MOTOR (Motor de execução dos genes)
# ══════════════════════════════════════════════════════════════════════

class GeneMotor:
    """
    Executa os genes de uma estratégia composta.
    Recebe o dicionário de parâmetros (p) com os genes selecionados
    e implementa: sinal_entrada, filtro_tendencia, filtro_volatilidade,
    atualizar_estado_smc.
    """

    def __init__(self, p: dict):
        self.p = p
        self.gene_entrada = p.get("gene_entrada", "CHoCH_FVG")
        self.gene_filtro_t = p.get("gene_filtro_t", "NENHUM")
        self.gene_filtro_v = p.get("gene_filtro_v", "NENHUM")
        self.gene_saida = p.get("gene_saida", "RR_FIXO")
        self._last_choch_dir = 0
        self._last_bos_dir = 0

    def atualizar_estado_smc(self, df, i):
        try:
            if "choch" in df.columns:
                v = df["choch"].iloc[i]
                if v != 0 and not (isinstance(v, float) and np.isnan(v)):
                    self._last_choch_dir = int(v)
            if "bos" in df.columns:
                v = df["bos"].iloc[i]
                if v != 0 and not (isinstance(v, float) and np.isnan(v)):
                    self._last_bos_dir = int(v)
        except Exception:
            pass

    def _poi_from_row(self, row, spread=5.0):
        c = row["close"]
        hi = row.get("high", c + spread)
        lo = row.get("low", c - spread)
        return {"top": float(hi), "bot": float(lo)}

    def sinal_entrada(self, df, i):
        ge = self.gene_entrada
        try:
            if ge == "CHoCH_FVG":       return self._e_choch_fvg(df, i)
            elif ge == "CHoCH_OB":      return self._e_choch_ob(df, i)
            elif ge == "CHoCH_FVG_OB":
                d, poi, st = self._e_choch_fvg(df, i)
                return (d, poi, st) if d != 0 else self._e_choch_ob(df, i)
            elif ge == "LIQ_SWEEP":     return self._e_liq_sweep(df, i)
            elif ge == "BREAKOUT_VOL":  return self._e_breakout_vol(df, i)
            elif ge == "RSI_EXTREME":   return self._e_rsi_extreme(df, i)
            elif ge == "EMA_CROSS":     return self._e_ema_cross(df, i)
            elif ge == "BB_REVERSAL":   return self._e_bb_reversal(df, i)
            elif ge == "MACD_SIGNAL":   return self._e_macd_signal(df, i)
            elif ge == "DOJI_REVERSAL": return self._e_doji_reversal(df, i)
            elif ge == "ENGULF_SMC":    return self._e_engulf_smc(df, i)
            elif ge == "MOMENTUM_BREAK":return self._e_momentum_break(df, i)
        except Exception:
            pass
        return 0, None, "ERRO"

    def _e_choch_fvg(self, df, i):
        if "choch" not in df.columns:
            return 0, None, "NO_COL"
        janela = self.p.get("choch_janela", 50)
        start = max(0, i - janela)
        last_choch = 0
        for v in reversed(df["choch"].iloc[start:i+1].values):
            if v != 0 and not (isinstance(v, float) and np.isnan(v)):
                last_choch = int(v); break
        if last_choch == 0:
            return 0, None, "NO_CHOCH"
        fvg_col = "fvg_up" if last_choch == 1 else "fvg_dn"
        if fvg_col in df.columns and not df[fvg_col].iloc[start:i+1].any():
            return 0, None, "NO_FVG"
        return last_choch, self._poi_from_row(df.iloc[i]), "CHoCH_FVG"

    def _e_choch_ob(self, df, i):
        if "choch" not in df.columns:
            return 0, None, "NO_COL"
        janela = self.p.get("choch_janela", 50)
        start = max(0, i - janela)
        last_choch = 0
        for v in reversed(df["choch"].iloc[start:i+1].values):
            if v != 0 and not (isinstance(v, float) and np.isnan(v)):
                last_choch = int(v); break
        if last_choch == 0:
            return 0, None, "NO_CHOCH"
        ob_col = "ob_bull" if last_choch == 1 else "ob_bear"
        if ob_col in df.columns and not df[ob_col].iloc[start:i+1].any():
            return 0, None, "NO_OB"
        return last_choch, self._poi_from_row(df.iloc[i]), "CHoCH_OB"

    def _e_liq_sweep(self, df, i):
        janela = self.p.get("swing_length", 5)
        start = max(0, i - janela * 3)
        sub = df.iloc[start:i]
        if len(sub) < janela:
            return 0, None, "INSUF"
        sh = sub["high"].max()
        sl = sub["low"].min()
        hi_i = df["high"].iloc[i]
        lo_i = df["low"].iloc[i]
        cl = df["close"].iloc[i]
        if hi_i > sh and cl < sh:
            return -1, {"top": float(hi_i), "bot": float(sh)}, "LIQ_BEAR"
        if lo_i < sl and cl > sl:
            return 1, {"top": float(sl), "bot": float(lo_i)}, "LIQ_BULL"
        return 0, None, "NO_SWEEP"

    def _e_breakout_vol(self, df, i):
        p = self.p
        per = p.get("breakout_period", 20)
        start = max(0, i - per)
        sub = df.iloc[start:i]
        if len(sub) < per // 2:
            return 0, None, "INSUF"
        max_h = sub["high"].max()
        min_l = sub["low"].min()
        cl = df["close"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_ma = df["vol_ma"].iloc[i] if "vol_ma" in df.columns else vol
        if np.isnan(vol_ma) or vol_ma == 0:
            return 0, None, "NO_VOLMA"
        ok = vol >= vol_ma * p.get("vol_mult", 1.5)
        row = df.iloc[i]
        if cl > max_h and ok:
            return 1, {"top": float(cl), "bot": float(max_h)}, "BREAK_BULL"
        if cl < min_l and ok:
            return -1, {"top": float(min_l), "bot": float(cl)}, "BREAK_BEAR"
        return 0, None, "NO_BREAK"

    def _e_rsi_extreme(self, df, i):
        if "rsi" not in df.columns:
            return 0, None, "NO_RSI"
        rsi = df["rsi"].iloc[i]
        if np.isnan(rsi):
            return 0, None, "NAN"
        p = self.p
        row = df.iloc[i]
        if rsi <= p.get("rsi_low", 30):
            return 1, self._poi_from_row(row), "RSI_OS"
        if rsi >= p.get("rsi_high", 70):
            return -1, self._poi_from_row(row), "RSI_OB"
        return 0, None, "RSI_NEUT"

    def _e_ema_cross(self, df, i):
        if i < 2 or "ema_fast" not in df.columns or "ema_slow" not in df.columns:
            return 0, None, "NO_EMA"
        ef0, es0 = df["ema_fast"].iloc[i], df["ema_slow"].iloc[i]
        ef1, es1 = df["ema_fast"].iloc[i-1], df["ema_slow"].iloc[i-1]
        if any(np.isnan(v) for v in [ef0, es0, ef1, es1]):
            return 0, None, "NAN"
        row = df.iloc[i]
        if ef1 <= es1 and ef0 > es0:
            return 1, self._poi_from_row(row), "EMA_GOLDEN"
        if ef1 >= es1 and ef0 < es0:
            return -1, self._poi_from_row(row), "EMA_DEATH"
        return 0, None, "NO_CROSS"

    def _e_bb_reversal(self, df, i):
        if "bb_up" not in df.columns or "bb_lo" not in df.columns:
            return 0, None, "NO_BB"
        cl = df["close"].iloc[i]
        bu = df["bb_up"].iloc[i]
        bl = df["bb_lo"].iloc[i]
        if np.isnan(bu) or np.isnan(bl):
            return 0, None, "NAN"
        row = df.iloc[i]
        if cl <= bl:
            return 1, self._poi_from_row(row), "BB_LOW"
        if cl >= bu:
            return -1, self._poi_from_row(row), "BB_HIGH"
        return 0, None, "BB_NEUT"

    def _e_macd_signal(self, df, i):
        if i < 2 or "macd" not in df.columns or "macd_sig" not in df.columns:
            return 0, None, "NO_MACD"
        m0, s0 = df["macd"].iloc[i], df["macd_sig"].iloc[i]
        m1, s1 = df["macd"].iloc[i-1], df["macd_sig"].iloc[i-1]
        if any(np.isnan(v) for v in [m0, s0, m1, s1]):
            return 0, None, "NAN"
        row = df.iloc[i]
        if m1 <= s1 and m0 > s0:
            return 1, self._poi_from_row(row), "MACD_BULL"
        if m1 >= s1 and m0 < s0:
            return -1, self._poi_from_row(row), "MACD_BEAR"
        return 0, None, "MACD_NEUT"

    def _e_doji_reversal(self, df, i):
        if i < 3:
            return 0, None, "INSUF"
        row = df.iloc[i]
        corpo = abs(row["close"] - row["open"])
        sombra = row["high"] - row["low"]
        if sombra == 0 or corpo / sombra > 0.25:
            return 0, None, "NAO_DOJI"
        closes = df["close"].iloc[i-3:i]
        trend = closes.iloc[-1] - closes.iloc[0]
        poi = self._poi_from_row(row)
        if trend < 0:
            return 1, poi, "DOJI_BULL"
        if trend > 0:
            return -1, poi, "DOJI_BEAR"
        return 0, None, "DOJI_NEUT"

    def _e_engulf_smc(self, df, i):
        if i < 2:
            return 0, None, "INSUF"
        prev, curr = df.iloc[i-1], df.iloc[i]
        po, pc = prev["open"], prev["close"]
        co, cc = curr["open"], curr["close"]
        row = df.iloc[i]
        if pc < po and cc > co and cc > po and co < pc:
            if "choch" in df.columns:
                rc = df["choch"].iloc[max(0,i-30):i]
                if not any(v == 1 for v in rc if not (isinstance(v,float) and np.isnan(v))):
                    return 0, None, "NO_SMC"
            return 1, self._poi_from_row(row), "ENGULF_BULL"
        if pc > po and cc < co and cc < po and co > pc:
            if "choch" in df.columns:
                rc = df["choch"].iloc[max(0,i-30):i]
                if not any(v == -1 for v in rc if not (isinstance(v,float) and np.isnan(v))):
                    return 0, None, "NO_SMC"
            return -1, self._poi_from_row(row), "ENGULF_BEAR"
        return 0, None, "NO_ENGULF"

    def _e_momentum_break(self, df, i):
        if "roc" not in df.columns:
            return 0, None, "NO_ROC"
        roc = df["roc"].iloc[i]
        if np.isnan(roc):
            return 0, None, "NAN"
        thr = self.p.get("roc_threshold", 0.5)
        row = df.iloc[i]
        if roc >= thr:
            return 1, self._poi_from_row(row), "MOM_BULL"
        if roc <= -thr:
            return -1, self._poi_from_row(row), "MOM_BEAR"
        return 0, None, "MOM_NEUT"

    # ------------------------------------------------------------------
    # FILTROS
    # ------------------------------------------------------------------

    def filtro_tendencia(self, row, direcao: int) -> bool:
        gt = self.gene_filtro_t
        if gt == "NENHUM":
            return True
        try:
            if gt == "EMA_FAST_SLOW":
                ef, es = row.get("ema_fast", np.nan), row.get("ema_slow", np.nan)
                if np.isnan(ef) or np.isnan(es): return True
                return (direcao == 1 and ef > es) or (direcao == -1 and ef < es)
            elif gt == "EMA_200":
                e200 = row.get("ema_200", np.nan)
                cl = row.get("close", np.nan)
                if np.isnan(e200) or np.isnan(cl): return True
                return (direcao == 1 and cl > e200) or (direcao == -1 and cl < e200)
            elif gt == "ADX_TREND":
                adx = row.get("adx", np.nan)
                if np.isnan(adx): return True
                return adx >= self.p.get("adx_threshold", 25)
            elif gt == "HH_HL":
                sh = row.get("swing_hi", np.nan)
                sl_ = row.get("swing_lo", np.nan)
                cl = row.get("close", np.nan)
                if np.isnan(sh) or np.isnan(sl_): return True
                return (direcao == 1 and cl > sl_) or (direcao == -1 and cl < sh)
            elif gt == "MACD_HIST":
                mh = row.get("macd_hist", np.nan)
                if np.isnan(mh): return True
                return (direcao == 1 and mh > 0) or (direcao == -1 and mh < 0)
            elif gt == "SUPERTREND":
                st = row.get("supertrend", np.nan)
                if np.isnan(st): return True
                return (direcao == 1 and st == 1) or (direcao == -1 and st == -1)
        except Exception:
            return True
        return True

    def filtro_volatilidade(self, row) -> bool:
        gv = self.gene_filtro_v
        if gv == "NENHUM":
            return True
        try:
            atr = row.get("atr", np.nan)
            atr_slow = row.get("atr_slow", np.nan)
            if gv == "ATR_EXPANDING":
                if np.isnan(atr) or np.isnan(atr_slow): return True
                return atr >= atr_slow * self.p.get("atr_expand_mult", 1.0)
            elif gv == "ATR_CONTRACTING":
                if np.isnan(atr) or np.isnan(atr_slow): return True
                return atr <= atr_slow * self.p.get("atr_contract_mult", 0.8)
            elif gv == "BB_SQUEEZE":
                bu = row.get("bb_up", np.nan)
                bl = row.get("bb_lo", np.nan)
                cl = row.get("close", 1.0)
                if np.isnan(bu) or np.isnan(bl) or cl == 0: return True
                return (bu - bl) / cl <= self.p.get("bb_squeeze_thr", 0.02)
            elif gv == "ATR_RANGE":
                if np.isnan(atr): return True
                return self.p.get("atr_range_lo", 2.0) <= atr <= self.p.get("atr_range_hi", 15.0)
            elif gv == "VOLUME_ABOVE_MA":
                vol = row.get("volume", np.nan)
                vm = row.get("vol_ma", np.nan)
                if np.isnan(vol) or np.isnan(vm) or vm == 0: return True
                return vol >= vm * self.p.get("vol_ma_mult", 1.2)
        except Exception:
            return True
        return True
