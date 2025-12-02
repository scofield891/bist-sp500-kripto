import os
from datetime import datetime
import requests

import yfinance as yf
import pandas as pd


# =============== Ayarlar ===============

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise RuntimeError("BOT_TOKEN ve CHAT_ID ortam değişkenlerini ayarla.")

TIMEFRAME_DAYS = "1d"  # Günlük mum

# Kripto tarafı ayarları (dinamik, marketcap top N)
TOP_CRYPTO_MC = 200  # Marketcap'e göre en büyük kaç coin taransın?


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


# =============== Kripto: Marketcap Top 200 (Yfinance Modu) ===============

def get_top_crypto_symbols_by_marketcap(limit: int = 200):
    """
    CoinGecko üzerinden, marketcap'e göre en büyük 'limit' coinin sembollerini çeker.
    Örn: ['BTC', 'ETH', 'USDT', 'BNB', ...] (BÜYÜK HARF).
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    symbols = []
    for coin in data:
        sym = str(coin.get("symbol", "")).upper().strip()
        if not sym:
            continue
        symbols.append(sym)

    return symbols


def scan_crypto_top_mcap(limit: int = 200):
    """
    1) CoinGecko'dan marketcap'e göre ilk 'limit' coini bulur.
    2) Bunları Yahoo Finance formatına (BTC-USD, ETH-USD) çevirir.
    3) Tek seferde toplu indirip EMA kesişimi arar.

    AVANTAJ: Rate limit derdi yok, çok hızlı, ccxt yok.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": [],
        "debug": ""
    }

    try:
        # 1) CoinGecko'dan listeyi çek
        try:
            cg_symbols = get_top_crypto_symbols_by_marketcap(limit=limit)
        except Exception as e:
            msg = f"CoinGecko hatası: {e}"
            print(msg)
            result["errors"].append(msg)
            return result

        # 2) Sembolleri Yahoo formatına çevir ve filtrele
        ignored_coins = [
            "USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDD", "USDP",
            "WBTC", "WETH", "STETH"
        ]

        yf_tickers = []
        original_map = {}  # YF sembolü -> Orijinal Coin sembolü

        for sym in cg_symbols:
            sym_u = sym.upper()
            if sym_u in ignored_coins:
                continue

            yf_sym = f"{sym_u}-USD"
            yf_tickers.append(yf_sym)
            original_map[yf_sym] = sym_u

        if not yf_tickers:
            result["debug"] = "Yahoo için uygun kripto sembolü bulunamadı."
            return result

        print(f"Kripto Taraması Başlıyor: {len(yf_tickers)} coin Yahoo Finance üzerinden çekiliyor...")

        # 3) Toplu İndirme
        try:
            data = yf.download(
                yf_tickers,
                period="400d",   # EMA için yeterli
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
        except Exception as e:
            msg = f"Yfinance toplu indirme hatası: {e}"
            print(msg)
            result["errors"].append(msg)
            return result

        multi = isinstance(data.columns, pd.MultiIndex)
        processed_count = 0

        # Tek coin durumu
        if not multi and len(yf_tickers) == 1:
            single_sym = yf_tickers[0]
            try:
                close = data["Close"].dropna()
                if not close.empty and len(close) >= 50:
                    display_name = original_map.get(single_sym, single_sym)
                    if has_recent_bullish_cross(close, 13, 34):
                        result["13_34_bull"].append(display_name)
                    if has_recent_bullish_cross(close, 34, 89):
                        result["34_89_bull"].append(display_name)
                    processed_count = 1
            except Exception as e:
                print("Tek kripto veri hatası:", e)
        else:
            # Çoklu sembol
            for yf_sym in yf_tickers:
                try:
                    if multi:
                        if yf_sym not in data.columns.levels[0]:
                            # Yahoo'da olmayan sembol, geç
                            continue
                        df_sym = data[yf_sym]
                    else:
                        # Beklenmedik yapı, atla
                        continue

                    if "Close" not in df_sym.columns:
                        continue

                    close = df_sym["Close"].dropna()
                    if close.empty or len(close) < 50:
                        continue

                    processed_count += 1

                    display_name = original_map.get(yf_sym, yf_sym)

                    if has_recent_bullish_cross(close, 13, 34):
                        result["13_34_bull"].append(display_name)

                    if has_recent_bullish_cross(close, 34, 89):
                        result["34_89_bull"].append(display_name)

                except Exception as e:
                    # Tek sembol hata verirse tüm akışı bozmasın
                    print("Kripto sembol hatası:", yf_sym, "->", e)
                    continue

        c13 = len(result["13_34_bull"])
        c34 = len(result["34_89_bull"])

        result["debug"] = (
            f"Kaynak: Yahoo Finance (Kripto). "
            f"Top mcap listesinden {len(yf_tickers)} coin denendi, "
            f"geçerli veri: {processed_count}. "
            f"Sinyaller -> 13/34: {c13} adet, 34/89: {c34} adet."
        )

    except Exception as e:
        result["errors"].append(f"Genel Kripto Hatası: {e}")

    return result


# =============== Formatlama ===============

def format_result_block(title: str, res: dict) -> str:
    lines = [f"📌 {title}"]

    def join_list(lst):
        return ", ".join(lst) if lst else "-"

    lines.append(f"EMA13-34 KESİŞİMİ : {join_list(res['13_34_bull'])}")
    lines.append(f"EMA34-89 KESİŞİMİ : {join_list(res['34_89_bull'])}")

    err_line = summarize_errors(res.get("errors", []))
    if err_line:
        lines.append(err_line)

    return "\n".join(lines)


# =============== Ana Akış ===============

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    header = (
        f"📊 EMA Yükseliş Kesişim Tarama – {today_str}\n"
        f"Timeframe: 1D (EMA13-34 & EMA34-89)\n"
        f"Evren: BIST 100, S&P 500, Global Kripto Top {TOP_CRYPTO_MC} (Marketcap, Yahoo Finance)\n"
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
    sp500_symbols = read_symbol_file("nasdaq100.txt")  # içine SP200 de koymuş olabilirsin, isim önemli değil
    if sp500_symbols:
        sp500_res = scan_equity_universe(sp500_symbols, "S&P 500")
        sp500_text = format_result_block("🇺🇸 S&P 500", sp500_res)
        send_telegram_message(sp500_text)

    # --- Kripto Top N (marketcap'e göre, dinamik, Yahoo Finance) --- #
    crypto_res = scan_crypto_top_mcap(limit=TOP_CRYPTO_MC)
    crypto_text = format_result_block(f"🪙 Kripto Top {TOP_CRYPTO_MC} (mcap, YF)", crypto_res)
    send_telegram_message(crypto_text)

    dbg = crypto_res.get("debug")
    if dbg:
        send_telegram_message("🔍 " + dbg)


if __name__ == "__main__":
    main()
