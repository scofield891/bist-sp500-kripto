import os
from datetime import datetime
import requests

import yfinance as yf
import pandas as pd
import ccxt
from dotenv import load_dotenv

# .env varsa lokal çalıştırırken de BOT_TOKEN / CHAT_ID gelsin
load_dotenv()

# =============== Ayarlar ===============

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("BOT_TOKEN ve CHAT_ID ortam değişkenlerini ayarla.")

TIMEFRAME_DAYS = "1d"  # Günlük mum (yfinance tarafı)

# ---- BIST evreni (havuz + likiditeye göre TOP N) ----
BIST_ALL_FILE = os.getenv("BIST_ALL_FILE", "bist_all.txt")
BIST_MAX_COUNT = int(os.getenv("BIST_MAX_COUNT", "150"))
BIST_LABEL = os.getenv("BIST_LABEL", f"BIST Top {BIST_MAX_COUNT} Likit")

# Kripto tarafı: Binance sembol listesi dosyası (BTC/USDT, ETH/USDT, ...)
BINANCE_LIST_FILE = os.getenv("BINANCE_LIST_FILE", "binance.txt")

# =============== Telegram ===============

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=20)
        if not r.ok:
            print("Telegram hata:", r.status_code, r.text)
    except Exception as e:
        print("Telegram gönderim hatası:", e)

# =============== Ortak Yardımcılar ===============

def read_symbol_file(path: str):
    """
    bist_all.txt / nasdaq100.txt / binance.txt gibi dosyalardan sembol listesi okur.
    Her satır 1 sembol: boş satırlar ve # ile başlayan satırlar atlanır.
    """
    if not os.path.exists(path):
        print(f"UYARI: {path} bulunamadı, bu evren taranmayacak.")
        return []

    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            symbols.append(line)
    return symbols


def select_most_liquid_bist_symbols(
    symbols,
    max_count: int = 150,
    lookback_days: int = 90,
    min_days: int = 30,
    universe_name: str = "BIST"
):
    """
    Verilen BIST sembolleri arasından, son 'lookback_days' içinde
    ortalama işlem değeri (Close * Volume) en yüksek olan ilk 'max_count'
    hisseyi seçer.
    """
    if not symbols:
        return []

    try:
        data = yf.download(
            symbols,
            period=f"{lookback_days}d",
            interval=TIMEFRAME_DAYS,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"{universe_name} likidite indirme hatası:", e)
        return symbols  # fallback

    multi = isinstance(data.columns, pd.MultiIndex)
    liquidity_list = []

    for sym in symbols:
        try:
            if multi:
                if sym not in data.columns.levels[0]:
                    continue
                df_sym = data[sym].dropna()
            else:
                df_sym = data

            if df_sym.empty:
                continue

            if "Close" not in df_sym.columns or "Volume" not in df_sym.columns:
                continue

            df_recent = df_sym.tail(60)
            if len(df_recent) < min_days:
                continue

            avg_value = (df_recent["Close"] * df_recent["Volume"]).mean()
            if pd.isna(avg_value) or avg_value <= 0:
                continue

            liquidity_list.append((sym, avg_value))

        except Exception as e:
            print(f"Likidite hesap hatası {sym}: {e}")
            continue

    if not liquidity_list:
        print(f"{universe_name} için likidite listesi boş, fallback ile tüm semboller kullanılacak.")
        return symbols

    liquidity_list.sort(key=lambda x: x[1], reverse=True)
    top_syms = [sym for sym, _ in liquidity_list[:max_count]]

    print(
        f"{universe_name}: {len(symbols)} sembolden likiditeye göre "
        f"ilk {len(top_syms)} seçildi (max_count={max_count})."
    )
    return top_syms


def has_recent_bullish_cross(
    close: pd.Series,
    fast: int,
    slow: int,
    max_bars_ago: int = 1,
    max_days_ago: int = 2,
    min_rel_gap: float = 0.0
) -> bool:
    """
    EMA fast & slow için bullish cross noktalarını bulur.

    Şartlar:
      - Cross, son bar veya en fazla max_bars_ago bar önce olacak.
      - Cross'un tarihi bugünden en fazla max_days_ago gün önce olacak.
      - Eğer min_rel_gap > 0 ise: cross barında EMA_fast - EMA_slow,
        fiyata oranla en az min_rel_gap olmalı.
    """
    if len(close) < slow + 3:
        return False

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    fast_above = ema_fast > ema_slow

    cross_indices = []
    for i in range(1, len(fast_above)):
        if fast_above.iloc[i] and not fast_above.iloc[i - 1]:
            cross_indices.append(i)

    if not cross_indices:
        return False

    last_cross = cross_indices[-1]
    last_idx = len(close) - 1

    # 1) Bar bazlı kontrol
    if last_cross < last_idx - max_bars_ago:
        return False

    # 1.5) Gap kontrolü
    if min_rel_gap > 0:
        try:
            gap = float(ema_fast.iloc[last_cross] - ema_slow.iloc[last_cross])
            price = float(close.iloc[last_cross])
            if price <= 0 or gap <= 0:
                return False
            if gap / price < min_rel_gap:
                return False
        except Exception as e:
            print("Gap kontrolü hatası (has_recent_bullish_cross):", e)
            return False

    # 2) Tarih bazlı kontrol
    idx = close.index
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            last_cross_time = idx[last_cross]

            if isinstance(last_cross_time, pd.Period):
                last_cross_time = last_cross_time.to_timestamp()

            if getattr(last_cross_time, "tzinfo", None) is not None:
                last_cross_time = last_cross_time.tz_convert("UTC").tz_localize(None)

            today_utc = pd.Timestamp.utcnow().normalize()
            cross_day = pd.Timestamp(last_cross_time).normalize()
            days_diff = (today_utc - cross_day).days

            if days_diff > max_days_ago:
                return False
        except Exception as e:
            print("Tarih kontrolü hatası (has_recent_bullish_cross):", e)

    return True


def summarize_errors(errors, max_show: int = 10) -> str:
    if not errors:
        return ""
    total = len(errors)
    if total <= max_show:
        return f"(Veri hatası: {', '.join(errors)})"
    shown = ", ".join(errors[:max_show])
    return f"(Veri hatası: {total} sembol, ilk {max_show}: {shown})"


# ===========================
# "Parayı Vurma Kesişimi"
# Close > AlphaTrend AND > R AND > FlowUpper
# Son 1-2 mum içinde False->True
# ===========================

def compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        (df["High"] - df["Low"]),
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr

def compute_mfi(df: pd.DataFrame, length: int) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df["Volume"].fillna(0)
    direction = tp.diff()
    pos = rmf.where(direction > 0, 0.0)
    neg = rmf.where(direction < 0, 0.0).abs()
    pos_sum = pos.rolling(length).sum()
    neg_sum = neg.rolling(length).sum()
    mfr = pos_sum / neg_sum.replace(0, pd.NA)
    mfi = 100 - (100 / (1 + mfr))
    return mfi.astype(float)

def compute_rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def compute_alpha_trend(df: pd.DataFrame, AP: int = 14, coeff: float = 1.0, novolumedata: bool = False) -> pd.Series:
    tr = compute_true_range(df)
    ATR = tr.rolling(AP).mean()  # ta.sma(ta.tr, AP)
    upT = df["Low"]  - ATR * coeff
    dnT = df["High"] + ATR * coeff

    # mfi >= 50 (volume yoksa rsi >= 50)
    has_vol = df["Volume"].fillna(0).gt(0).any()
    if novolumedata or (not has_vol):
        mom = compute_rsi(df["Close"], AP)
    else:
        mom = compute_mfi(df, AP)

    cond = mom >= 50

    alpha = [None] * len(df)
    for i in range(len(df)):
        prev = alpha[i-1] if i > 0 else None
        prev_val = prev if prev is not None else None

        if prev_val is None:
            alpha[i] = float(upT.iloc[i]) if bool(cond.iloc[i]) else float(dnT.iloc[i])
        else:
            if bool(cond.iloc[i]):
                alpha[i] = float(prev_val) if float(upT.iloc[i]) < float(prev_val) else float(upT.iloc[i])
            else:
                alpha[i] = float(prev_val) if float(dnT.iloc[i]) > float(prev_val) else float(dnT.iloc[i])

    return pd.Series(alpha, index=df.index, dtype=float)

def compute_altcointurk_R(df: pd.DataFrame, n1: int = 21, act: int = 21) -> pd.Series:
    hh = df["High"].rolling(n1).max()
    ll = df["Low"].rolling(n1).min()
    mid = (hh + ll) / 2.0
    R = mid.rolling(act).mean()  # ta.sma
    return R

def compute_flow_upper(df: pd.DataFrame, flowLen: int = 34) -> pd.Series:
    # Flow band upper = ta.ema(high, len)
    return df["High"].ewm(span=flowLen, adjust=False).mean()

def has_recent_parayi_vurma_kesisimi(
    df: pd.DataFrame,
    max_bars_ago: int = 1,
    max_days_ago: int = 2,
    AP: int = 14,
    coeff: float = 1.0,
    n1: int = 21,
    act: int = 21,
    flowLen: int = 34
) -> bool:
    if df is None or df.empty:
        return False
    need = max(200, flowLen + 10, n1 + act + 10, AP + 50)
    if len(df) < need:
        return False

    df = df.dropna().copy()
    if df.empty or len(df) < need:
        return False

    at = compute_alpha_trend(df, AP=AP, coeff=coeff, novolumedata=False)
    R = compute_altcointurk_R(df, n1=n1, act=act)
    flowU = compute_flow_upper(df, flowLen=flowLen)

    close = df["Close"]
    cond = (close > at) & (close > R) & (close > flowU)

    # False -> True kesişimi
    cross_idx = []
    for i in range(1, len(cond)):
        if bool(cond.iloc[i]) and (not bool(cond.iloc[i-1])):
            cross_idx.append(i)

    if not cross_idx:
        return False

    last_cross = cross_idx[-1]
    last_i = len(cond) - 1

    # Bar kontrolü
    if last_cross < last_i - max_bars_ago:
        return False

    # Tarih kontrolü (yfinance tarafı için)
    idx = df.index
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            t = idx[last_cross]
            if isinstance(t, pd.Period):
                t = t.to_timestamp()
            if getattr(t, "tzinfo", None) is not None:
                t = t.tz_convert("UTC").tz_localize(None)
            today_utc = pd.Timestamp.utcnow().normalize()
            cross_day = pd.Timestamp(t).normalize()
            if (today_utc - cross_day).days > max_days_ago:
                return False
        except Exception:
            pass

    return True


# =============== Hisse Taraması (BIST & S&P 500, toplu yfinance) ===============

def scan_equity_universe(symbols, universe_name: str):
    """
    yfinance ile TÜM sembolleri toplu indirip,
    EMA 13-34, EMA 34-89 bullish cross + Parayı Vurma Kesişimi arar.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "parayi_vurma": [],
        "errors": []
    }

    if not symbols:
        return result

    try:
        data = yf.download(
            symbols,
            period="400d",            # <- son 400 günlük veri
            interval=TIMEFRAME_DAYS,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"{universe_name} toplu indirme hatası:", e)
        result["errors"].extend(symbols)
        return result

    multi = isinstance(data.columns, pd.MultiIndex)

    for sym in symbols:
        try:
            if multi:
                if sym not in data.columns.levels[0]:
                    result["errors"].append(sym)
                    continue
                df_sym = data[sym].dropna()
            else:
                df_sym = data.dropna()

            needed_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not needed_cols.issubset(set(df_sym.columns)):
                result["errors"].append(sym)
                continue

            close = df_sym["Close"].dropna()
            if close.empty:
                result["errors"].append(sym)
                continue

            # EMA Crosslar
            if has_recent_bullish_cross(close, 13, 34):
                result["13_34_bull"].append(sym)

            if has_recent_bullish_cross(close, 34, 89):
                result["34_89_bull"].append(sym)

            # Parayı Vurma Kesişimi
            if has_recent_parayi_vurma_kesisimi(df_sym, max_bars_ago=1, max_days_ago=2):
                result["parayi_vurma"].append(sym)

        except Exception as e:
            print(f"{universe_name} veri hatası {sym}: {e}")
            result["errors"].append(sym)

    return result


# =============== Kripto: Binance listesi, MEXC 1D ===============

CRYPTO_TIMEFRAME = "1d"
CRYPTO_OHLC_LIMIT = 420  # PVK için uzun veri iyi olur

def find_mexc_symbol(binance_symbol: str, markets: dict):
    s = binance_symbol.strip().upper()
    if not s:
        return None

    if "/" in s:
        base, quote = s.split("/")
    else:
        if s.endswith("USDT"):
            base, quote = s[:-4], "USDT"
        else:
            base, quote = s, "USDT"

    candidates = [
        f"{base}/{quote}",
        f"{base}/{quote}:USDT",
    ]

    for c in candidates:
        if c in markets:
            return c

    return None


def scan_crypto_from_mexc_list() -> dict:
    """
    binance.txt içindeki sembolleri alır,
    MEXC 1D OHLCV'den EMA 13-34 / 34-89 bullish cross + Parayı Vurma Kesişimi tarar.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "parayi_vurma": [],
        "errors": [],
        "debug": ""
    }

    symbols = read_symbol_file(BINANCE_LIST_FILE)
    if not symbols:
        result["debug"] = f"{BINANCE_LIST_FILE} boş veya bulunamadı."
        return result

    try:
        exchange = ccxt.mexc({
            "enableRateLimit": True,
        })
        markets = exchange.load_markets()
    except Exception as e:
        msg = f"MEXC borsası başlatılamadı: {e}"
        print(msg)
        result["errors"].append(msg)
        return result

    processed_count = 0
    have_market_count = 0

    for sym in symbols:
        raw_sym = sym.strip()
        if not raw_sym:
            continue

        mexc_symbol = find_mexc_symbol(raw_sym, markets)
        if mexc_symbol is None:
            msg = f"{raw_sym}: MEXC'te uygun market bulunamadı"
            print(msg)
            result["errors"].append(msg)
            continue

        have_market_count += 1

        try:
            ohlcv = exchange.fetch_ohlcv(
                mexc_symbol,
                timeframe=CRYPTO_TIMEFRAME,
                limit=CRYPTO_OHLC_LIMIT,
            )
        except Exception as e:
            msg = f"{raw_sym} ({mexc_symbol}): {e}"
            print("Kripto veri hatası:", msg)
            result["errors"].append(msg)
            continue

        if not ohlcv or len(ohlcv) < 120:
            msg = f"{raw_sym} ({mexc_symbol}): yetersiz OHLCV verisi"
            print(msg)
            result["errors"].append(msg)
            continue

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "Open", "High", "Low", "Close", "Volume"]
        )

        close = df["Close"].astype(float)
        display_name = raw_sym.replace("/USDT", "")

        # EMA Crosslar
        if has_recent_bullish_cross(close, 13, 34, min_rel_gap=0.0, max_bars_ago=1, max_days_ago=9999):
            result["13_34_bull"].append(display_name)

        if has_recent_bullish_cross(close, 34, 89, min_rel_gap=0.0, max_bars_ago=1, max_days_ago=9999):
            result["34_89_bull"].append(display_name)

        # Parayı Vurma Kesişimi (kripto: tarih kontrolü yok, bar bazlı)
        if has_recent_parayi_vurma_kesisimi(df, max_bars_ago=1, max_days_ago=9999):
            result["parayi_vurma"].append(display_name)

        processed_count += 1

    c13 = len(result["13_34_bull"])
    c34 = len(result["34_89_bull"])
    cpv = len(result["parayi_vurma"])

    result["debug"] = (
        f"Kaynak: MEXC 1D. Binance listesinden {len(symbols)} satır okundu, "
        f"MEXC'te market bulunan: {have_market_count}, "
        f"geçerli veri çekilen: {processed_count}. "
        f"Sinyaller -> 13/34: {c13} adet, 34/89: {c34} adet, Parayı Vurma: {cpv} adet."
    )

    return result


# =============== Formatlama ===============

def format_result_block(title: str, res: dict) -> str:
    lines = [f"📌 {title}"]

    def join_list(lst):
        return ", ".join(lst) if lst else "-"

    lines.append(f"EMA13-34 KESİŞİMİ : {join_list(res.get('13_34_bull', []))}")
    lines.append(f"EMA34-89 KESİŞİMİ : {join_list(res.get('34_89_bull', []))}")
    lines.append(f"💥 Parayı Vurma Kesişimi : {join_list(res.get('parayi_vurma', []))}")

    err_line = summarize_errors(res.get("errors", []))
    if err_line:
        lines.append(err_line)

    return "\n".join(lines)


# =============== Ana Akış ===============

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    header = (
        f"📊 Tarama – {today_str}\n"
        f"Timeframe: 1D\n"
        f"Evren: {BIST_LABEL}, S&P 500, Seçili Kripto (MEXC 1D)\n"
        f"Modüller:\n"
        f"  • EMA13-34 & EMA34-89 (bullish cross)\n"
        f"  • 💥 Parayı Vurma Kesişimi (Close > AlphaTrend + R + FlowUpper)\n"
        f"NOT: Sadece son 1 mumda (en fazla 2 gün) oluşan sinyaller listelenir."
    )
    send_telegram_message(header)

    # --- BIST (havuzdan likiditeye göre TOP N) --- #
    bist_all = read_symbol_file(BIST_ALL_FILE)

    if bist_all:
        bist_symbols = select_most_liquid_bist_symbols(
            bist_all,
            max_count=BIST_MAX_COUNT,
            universe_name="BIST Likit"
        )

        if bist_symbols:
            bist_res = scan_equity_universe(bist_symbols, "BIST Likit")
            bist_label_full = f"{BIST_LABEL} ({len(bist_symbols)} hisse)"
            bist_text = format_result_block(f"🇹🇷 {bist_label_full}", bist_res)
            send_telegram_message(bist_text)
    else:
        print(f"{BIST_ALL_FILE} bulunamadı, BIST taraması yapılmayacak.")

    # --- S&P 500 (dosyadan okunuyor: nasdaq100.txt) --- #
    sp500_symbols = read_symbol_file("nasdaq100.txt")
    if sp500_symbols:
        sp500_res = scan_equity_universe(sp500_symbols, "S&P 500")
        sp500_text = format_result_block("🇺🇸 S&P 500", sp500_res)
        send_telegram_message(sp500_text)

    # --- Kripto (Binance listesi, MEXC 1D) --- #
    crypto_res = scan_crypto_from_mexc_list()
    crypto_text = format_result_block("🪙 Kripto (Binance listesi, MEXC 1D)", crypto_res)
    send_telegram_message(crypto_text)

    dbg = crypto_res.get("debug")
    if dbg:
        send_telegram_message("🔍 " + dbg)


if __name__ == "__main__":
    main()
