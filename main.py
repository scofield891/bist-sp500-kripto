import os
from datetime import datetime
import requests

import ccxt
import yfinance as yf
import pandas as pd


# =============== Ortam Değişkenleri (GitHub Secrets'ten gelecek) ===============

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("BOT_TOKEN ve CHAT_ID ortam değişkenlerini ayarla.")


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

TIMEFRAME_DAYS = "1d"  # Günlük mum

def read_symbol_file(path: str):
    """
    bist100.txt / nasdaq100.txt gibi dosyalardan sembol listesi okur.
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


def compute_bullish_cross(close: pd.Series, fast: int, slow: int) -> bool:
    """
    EMA fast & slow için sadece YUKARI kesişime bakar.
    Dönüş:
        True  -> bullish (yukarı kesişim var)
        False -> yok
    """
    if len(close) < slow + 2:
        return False

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    prev_fast, last_fast = ema_fast.iloc[-2], ema_fast.iloc[-1]
    prev_slow, last_slow = ema_slow.iloc[-2], ema_slow.iloc[-1]

    # Aşağıdan yukarı kesti mi?
    return prev_fast < prev_slow and last_fast > last_slow


# =============== Hisse Taraması (BIST & Nasdaq) ===============

def scan_equity_universe(symbols, universe_name: str):
    """
    yfinance ile günlük veri çekip sadece bullish
    EMA 13-34 ve EMA 34-89 kesişimlerini bulur.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": []
    }

    for sym in symbols:
        try:
            data = yf.download(
                sym,
                period="300d",
                interval=TIMEFRAME_DAYS,
                auto_adjust=False,
                progress=False
            )
            if data is None or data.empty:
                result["errors"].append(sym)
                continue

            close = data["Close"].dropna()
            if close.empty:
                result["errors"].append(sym)
                continue

            # EMA 13-34 bullish cross
            if compute_bullish_cross(close, 13, 34):
                result["13_34_bull"].append(sym)

            # EMA 34-89 bullish cross
            if compute_bullish_cross(close, 34, 89):
                result["34_89_bull"].append(sym)

        except Exception as e:
            print(f"{universe_name} veri hatası {sym}: {e}")
            result["errors"].append(sym)

    return result


# =============== Binance Spot USDT Taraması ===============

def get_binance_spot_usdt_symbols():
    """
    ccxt ile Binance'den tüm spot USDT paritelerini çeker.
    Futures kullanılmaz, sadece spot.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    markets = exchange.load_markets()
    symbols = []

    for m in markets.values():
        try:
            if not m.get("spot"):
                continue
            if m.get("quote") != "USDT":
                continue
            symbols.append(m["symbol"])
        except Exception:
            continue

    symbols = sorted(list(set(symbols)))
    print(f"Binance spot USDT sembol sayısı: {len(symbols)}")
    return exchange, symbols


def scan_binance_spot_usdt(exchange, symbols):
    """
    Binance spot USDT marketler için sadece bullish
    EMA 13-34 ve EMA 34-89 kesişimleri.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": []
    }

    for sym in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(sym, timeframe="1d", limit=150)
            if not ohlcv or len(ohlcv) < 50:
                result["errors"].append(sym)
                continue

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            close = df["close"].dropna()

            if compute_bullish_cross(close, 13, 34):
                result["13_34_bull"].append(sym)

            if compute_bullish_cross(close, 34, 89):
                result["34_89_bull"].append(sym)

        except Exception as e:
            print("Binance hatası", sym, ":", e)
            result["errors"].append(sym)

    return result


# =============== Formatlama ===============

def format_result_block(title: str, res: dict) -> str:
    lines = [f"📌 {title}"]

    def join_list(lst):
        return ", ".join(lst) if lst else "-"

    lines.append(f"  EMA13-34 YUKARI KESENLER : {join_list(res['13_34_bull'])}")
    lines.append(f"  EMA34-89 YUKARI KESENLER : {join_list(res['34_89_bull'])}")

    if res["errors"]:
        lines.append(f"  (Veri hatası: {', '.join(res['errors'])})")

    return "\n".join(lines)


# =============== Ana Akış ===============

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    header = (
        f"📊 EMA Yükseliş Kesişim Tarama – {today_str}\n"
        f"Timeframe: 1D (EMA13-34 & EMA34-89)\n"
        f"Evren: BIST 100, Nasdaq 100, Binance Spot USDT\n"
        f"NOT: Sadece bullish (yukarı kesişim) sinyalleri listelenir."
    )
    send_telegram_message(header)

    # --- BIST 100 --- #
    bist_symbols = read_symbol_file("bist100.txt")
    if bist_symbols:
        bist_res = scan_equity_universe(bist_symbols, "BIST 100")
        bist_text = format_result_block("🇹🇷 BIST 100", bist_res)
        send_telegram_message(bist_text)

    # --- Nasdaq 100 --- #
    nasdaq_symbols = read_symbol_file("nasdaq100.txt")
    if nasdaq_symbols:
        nasdaq_res = scan_equity_universe(nasdaq_symbols, "Nasdaq 100")
        nasdaq_text = format_result_block("🇺🇸 Nasdaq 100", nasdaq_res)
        send_telegram_message(nasdaq_text)

    # --- Binance Spot USDT --- #
    try:
        exchange, binance_symbols = get_binance_spot_usdt_symbols()
        binance_res = scan_binance_spot_usdt(exchange, binance_symbols)
        binance_text = format_result_block("🪙 Binance Spot USDT", binance_res)
        send_telegram_message(binance_text)
    except Exception as e:
        send_telegram_message(f"⚠️ Binance spot USDT taramasında hata: {e}")


if __name__ == "__main__":
    main()
