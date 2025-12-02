import os
from datetime import datetime
import time
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

# Kripto tarafı ayarları (dinamik, marketcap top N)
TOP_CRYPTO_MC = 200            # Marketcap'e göre en büyük kaç coin taransın?
CRYPTO_EXCHANGE = "bybit"      # Binance sıkıntılı, Bybit kullanalım


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


# =============== Kripto: Marketcap'e göre TOP 200 (dinamik, Bybit) ===============

def get_top_crypto_symbols_by_marketcap(limit: int = 200):
    """
    CoinGecko üzerinden, marketcap'e göre en büyük 'limit' coinin sembollerini çeker.
    Örn: ['BTC', 'ETH', 'USDT', 'BNB', ...]
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


def map_to_exchange_symbol(sym: str, exchange):
    """
    CoinGecko sembolünü (BTC, ETH, SOL vs.)
    seçtiğimiz borsanın sembol formatına çevirir.
    Bybit için genelde 'BTC/USDT', 'SOL/USDT' gibi.
    Stablecoin'leri (USDT, USDC vs.) atlıyoruz.
    """
    s = sym.upper()

    # Stablecoin'leri direkt atla
    if s in ["USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDD", "USDP"]:
        return None

    markets = exchange.markets if hasattr(exchange, "markets") else exchange.load_markets()

    pair1 = s + "/USDT"
    pair2 = s + "/USDC"

    if pair1 in markets:
        return pair1
    if pair2 in markets:
        return pair2

    return None


def scan_crypto_top_mcap(limit: int = 200):
    """
    Marketcap'e göre en büyük 'limit' coini bulur (CoinGecko),
    seçili borsadan (CRYPTO_EXCHANGE) günlük OHLCV çekip
    EMA 13-34 ve 34-89 bullish cross taraması yapar.

    Pencere: son 1–2 mum (has_recent_bullish_cross ile aynı mantık).
    RateLimitExceeded yaşamamak için her istekte küçük sleep koyuyoruz.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": [],
        "debug": ""
    }

    try:
        # 1) CoinGecko'dan marketcap top listesi
        try:
            cg_symbols = get_top_crypto_symbols_by_marketcap(limit=limit)
        except Exception as e:
            msg = f"CoinGecko top list hatası: {type(e).__name__}"
            print(msg, e)
            result["errors"].append(msg)
            result["debug"] = "CoinGecko'dan marketcap listesi alınamadı."
            return result

        cg_count = len(cg_symbols)

        # 2) Borsaya bağlan (ccxt)
        try:
            exchange_class = getattr(ccxt, CRYPTO_EXCHANGE)
        except AttributeError:
            err = f"Geçersiz borsa ismi: {CRYPTO_EXCHANGE}"
            print(err)
            result["errors"].append(err)
            result["debug"] = err
            return result

        exchange = exchange_class({'enableRateLimit': True})
        markets = exchange.load_markets()

        # 3) CoinGecko sembollerini borsa sembolüne map et
        mapped_symbols = []
        not_listed = []

        for sym in cg_symbols:
            ex_sym = map_to_exchange_symbol(sym, exchange)
            if ex_sym is None:
                not_listed.append(sym)
            else:
                mapped_symbols.append(ex_sym)

        # uniq yap
        mapped_symbols = list(dict.fromkeys(mapped_symbols))

        ok_ohlcv = 0

        # 4) EMA taraması
        # Bybit rate limit'e takılmamak için her istekte biraz bekle
        # exchange.rateLimit milisaniye cinsinden -> saniyeye çevir
        base_sleep = 0.3
        try:
            if getattr(exchange, "rateLimit", None):
                base_sleep = max(base_sleep, exchange.rateLimit / 1000.0 * 1.2)
        except Exception:
            pass

        for ex_sym in mapped_symbols:
            try:
                time.sleep(base_sleep)

                ohlcv = exchange.fetch_ohlcv(ex_sym, timeframe="1d", limit=220)
                if not ohlcv or len(ohlcv) < 50:
                    result["errors"].append(f"{ex_sym} (veri yok)")
                    continue

                ok_ohlcv += 1

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                close = df["close"].dropna()
                if close.empty:
                    result["errors"].append(f"{ex_sym} (close boş)")
                    continue

                # Kriptoda da 2 mumluk pencere (aynı fonksiyon)
                if has_recent_bullish_cross(close, 13, 34):
                    result["13_34_bull"].append(ex_sym)

                if has_recent_bullish_cross(close, 34, 89):
                    result["34_89_bull"].append(ex_sym)

            except Exception as e:
                # RateLimitExceeded dahil tüm hataları burada yakala
                print("Kripto hatası", ex_sym, ":", e)
                result["errors"].append(f"{ex_sym} (hata: {type(e).__name__})")
                # Rate limit durumunda döngüye devam edelim, sistem komple patlamasın
                continue

        err_count = len(result["errors"])
        c13 = len(result["13_34_bull"])
        c34 = len(result["34_89_bull"])

        result["debug"] = (
            f"Kripto debug -> CoinGecko top mcap sayısı: {cg_count}, "
            f"borsada map edilen: {len(mapped_symbols)}, "
            f"OHLCV başarı: {ok_ohlcv}, "
            f"13-34 sinyal: {c13}, 34-89 sinyal: {c34}, "
            f"hata: {err_count}, "
            f"borsada listelenmeyen (örnek): {', '.join(not_listed[:10])}"
        )

    except Exception as e:
        # En üst seviye güvenlik ağı: hiçbir şey dışarı taşmasın
        msg = f"genel kripto hatası: {type(e).__name__}"
        print(msg, e)
        result["errors"].append(msg)
        if not result["debug"]:
            result["debug"] = msg

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
        f"Evren: BIST 100, S&P 500, Global Kripto Top {TOP_CRYPTO_MC} (Marketcap)\n"
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

    # --- Kripto Top N (marketcap'e göre, dinamik, Bybit) --- #
    crypto_res = scan_crypto_top_mcap(limit=TOP_CRYPTO_MC)
    crypto_text = format_result_block(f"🪙 Kripto Top {TOP_CRYPTO_MC} (mcap, {CRYPTO_EXCHANGE})", crypto_res)
    send_telegram_message(crypto_text)

    dbg = crypto_res.get("debug")
    if dbg:
        send_telegram_message("🔍 " + dbg)


if __name__ == "__main__":
    main()
