import os
from datetime import datetime
import requests

import ccxt
import yfinance as yf
import pandas as pd


# =============== Ayarlar ===============

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("BOT_TOKEN ve CHAT_ID ortam değişkenlerini ayarla.")

TIMEFRAME_DAYS = "1d"  # Günlük mum
BYBIT_LIST_FILE = "binance.txt"  # Binance USDT listesi dosyası (Bybit spot için kullanılacak)


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
    bist100.txt / nasdaq100.txt / binance.txt gibi dosyalardan sembol listesi okur.
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


def has_recent_bullish_cross(close: pd.Series, fast: int, slow: int) -> bool:
    """
    EMA fast & slow için bullish cross noktalarını bulur.
    Son bullish cross, son mumda veya bir önceki mumdaysa True döner.
    Aksi halde False.

    Bullish cross: EMA_fast > EMA_slow durumunun False -> True'ya geçtiği bar.
    """
    if len(close) < slow + 3:
        return False

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    fast_above = ema_fast > ema_slow  # boolean seri

    cross_indices = []
    for i in range(1, len(fast_above)):
        if fast_above.iloc[i] and not fast_above.iloc[i - 1]:
            cross_indices.append(i)

    if not cross_indices:
        return False

    last_cross = cross_indices[-1]
    last_idx = len(close) - 1

    # Son mum (last_idx) veya ondan bir önceki mum (last_idx - 1) kabul
    return last_cross >= last_idx - 1


def summarize_errors(errors, max_show: int = 10) -> str:
    if not errors:
        return ""
    total = len(errors)
    if total <= max_show:
        return f"(Veri hatası: {', '.join(errors)})"
    shown = ", ".join(errors[:max_show])
    return f"(Veri hatası: {total} sembol, ilk {max_show}: {shown})"


# =============== Hisse Taraması (BIST & S&P 500, toplu yfinance) ===============

def scan_equity_universe(symbols, universe_name: str):
    """
    yfinance ile TÜM sembolleri toplu indirip,
    EMA 13-34 ve EMA 34-89 için son 1 mum (max 2 mum) bullish cross arar.
    Toplu indirme = daha az hata / rate limit.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": []
    }

    if not symbols:
        return result

    try:
        # Bir kerede tüm sembolleri indir
        data = yf.download(
            symbols,
            period="400d",
            interval=TIMEFRAME_DAYS,
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"{universe_name} toplu indirme hatası:", e)
        # Hepsini error'a at
        result["errors"].extend(symbols)
        return result

    multi = isinstance(data.columns, pd.MultiIndex)

    for sym in symbols:
        try:
            if multi:
                # Çoklu ticker'da her sembol ayrı kolon seviyesinde
                if sym not in data.columns.levels[0]:
                    result["errors"].append(sym)
                    continue
                df_sym = data[sym].dropna()
            else:
                # Tek sembol ise direkt DataFrame
                df_sym = data

            if "Close" not in df_sym.columns:
                result["errors"].append(sym)
                continue

            close = df_sym["Close"].dropna()
            if close.empty:
                result["errors"].append(sym)
                continue

            if has_recent_bullish_cross(close, 13, 34):
                result["13_34_bull"].append(sym)

            if has_recent_bullish_cross(close, 34, 89):
                result["34_89_bull"].append(sym)

        except Exception as e:
            print(f"{universe_name} veri hatası {sym}: {e}")
            result["errors"].append(sym)

    return result


# =============== Bybit Spot (Binance listesi üzerinden) ===============

def normalize_to_bybit_symbol(raw: str) -> str:
    """
    Binance formatındaki sembolü (BTCUSDT, BTCUSDT.P, BTCUSDT-PERP vb.)
    Bybit/ccxt formatına çevirir (BTC/USDT).
    Eğer zaten içinde '/' varsa olduğu gibi bırakır.
    """
    if not raw:
        return raw

    s = raw.strip().upper()

    # Zaten ccxt formatındaysa dokunma
    if "/" in s:
        return s

    # Futures / perpetual eklerini temizle
    for suf in [".P", "_P", "-PERP", "PERP"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break

    # Klasik USDT çifti
    if s.endswith("USDT") and len(s) > 4:
        base = s[:-4]
        return f"{base}/USDT"

    # USD çifti (gerekirse)
    if s.endswith("USD") and len(s) > 3:
        base = s[:-3]
        return f"{base}/USD"

    # Diğerleri için olduğu gibi dön
    return s


def scan_bybit_spot_from_file(path: str):
    """
    binance.txt içindeki coinleri,
    Bybit spot marketlerinde EMA 13-34 ve EMA 34-89 için son 1–2 mumda bullish cross açısından tarar.
    """
    symbols_raw = read_symbol_file(path)
    if not symbols_raw:
        return {
            "13_34_bull": [],
            "34_89_bull": [],
            "errors": []
        }

    bybit = ccxt.bybit({'enableRateLimit': True})
    markets = bybit.load_markets()

    # Sadece spot market sembolleri
    available_symbols = {
        symbol for symbol, m in markets.items()
        if m.get("spot")
    }

    print(f"Bybit spot sembol sayısı: {len(available_symbols)}")
    print(f"binance.txt sembol sayısı: {len(symbols_raw)}")

    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": []
    }

    for raw in symbols_raw:
        bybit_sym = normalize_to_bybit_symbol(raw)

        if bybit_sym not in available_symbols:
            result["errors"].append(f"{raw} (Bybit'te yok: {bybit_sym})")
            continue

        try:
            ohlcv = bybit.fetch_ohlcv(bybit_sym, timeframe="1d", limit=220)
            if not ohlcv or len(ohlcv) < 50:
                result["errors"].append(f"{raw} (veri yok)")
                continue

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            close = df["close"].dropna()
            if close.empty:
                result["errors"].append(f"{raw} (close boş)")
                continue

            if has_recent_bullish_cross(close, 13, 34):
                result["13_34_bull"].append(raw)

            if has_recent_bullish_cross(close, 34, 89):
                result["34_89_bull"].append(raw)

        except Exception as e:
            print("Bybit hatası", raw, "->", bybit_sym, ":", e)
            result["errors"].append(f"{raw} (hata: {type(e).__name__})")

    return result


# =============== Formatlama ===============

def format_result_block(title: str, res: dict) -> str:
    lines = [f"📌 {title}"]

    def join_list(lst):
        return ", ".join(lst) if lst else "-"

    lines.append(f"EMA13-34 YENİ/YAKIN KESİŞİM : {join_list(res['13_34_bull'])}")
    lines.append(f"EMA34-89 YENİ/YAKIN KESİŞİM : {join_list(res['34_89_bull'])}")

    err_line = summarize_errors(res["errors"])
    if err_line:
        lines.append(err_line)

    return "\n".join(lines)


# =============== Ana Akış ===============

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    header = (
        f"📊 EMA Yükseliş Kesişim Tarama – {today_str}\n"
        f"Timeframe: 1D (EMA13-34 & EMA34-89)\n"
        f"Evren: BIST 100, S&P 500, Bybit Spot (Binance USDT listesi)\n"
        f"NOT: Sadece son 1 mumda veya en fazla 2 mum önce oluşmuş bullish kesişimler listelenir."
    )
    send_telegram_message(header)

    # --- BIST 100 --- #
    bist_symbols = read_symbol_file("bist100.txt")
    if bist_symbols:
        bist_res = scan_equity_universe(bist_symbols, "BIST 100")
        bist_text = format_result_block("🇹🇷 BIST 100", bist_res)
        send_telegram_message(bist_text)

    # --- S&P 500 (nasdaq100.txt dosyasından okunuyor) --- #
    sp500_symbols = read_symbol_file("nasdaq100.txt")  # dosya adı şimdilik böyle
    if sp500_symbols:
        sp500_res = scan_equity_universe(sp500_symbols, "S&P 500")
        sp500_text = format_result_block("🇺🇸 S&P 500", sp500_res)
        send_telegram_message(sp500_text)

    # --- Bybit Spot (Binance listesinden) --- #
    bybit_res = scan_bybit_spot_from_file(BYBIT_LIST_FILE)
    bybit_text = format_result_block("🪙 Bybit Spot (Binance USDT listesi)", bybit_res)
    send_telegram_message(bybit_text)


if __name__ == "__main__":
    main()
