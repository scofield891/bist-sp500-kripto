import os
from datetime import datetime
import requests
import time

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

# Kripto hacim filtresi: 30 günlük ortalama günlük hacim minimum (USD)
CRYPTO_MIN_VOLUME_30D = float(os.getenv("CRYPTO_MIN_VOLUME_30D", "750000"))

# =============== Blacklist (Stablecoin + Fan Token) ===============

CRYPTO_BLACKLIST = {
    # Stablecoinler
    "USDC", "TUSD", "FDUSD", "USDE", "USDP", "USD1", "XUSD",
    "EURI", "EUR", "BUSD", "DAI", "PAXG", "GUSD", "USDJ",
    "USDD", "USTC", "TUSD", "AEUR",
    
    # Fan tokenler
    "BAR", "PSG", "SANTOS", "LAZIO", "PORTO", "ACM", "ASR",
    "CITY", "ALPINE", "OG", "JUV", "ATM", "INTER", "AFC",
    "NAV", "SPURS",
    
    # Wrapped tokenler (orijinali zaten listede)
    "WBTC", "WETH", "WBNB",
    
    # Dead/Ölü projeler
    "LUNC", "LUNA2",
}

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


# =============== Hisse Taraması (BIST & NASDAQ, toplu yfinance) ===============

def scan_equity_universe(symbols, universe_name: str):
    """
    yfinance ile TÜM sembolleri toplu indirip,
    EMA 13-34 için son 1 mum (max 2 mum) bullish cross arar.
    """
    result = {
        "13_34_bull": [],
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

        except Exception as e:
            print(f"{universe_name} veri hatası {sym}: {e}")
            result["errors"].append(sym)

    return result


# =============== Kripto: Binance Hacim + MEXC Mum Verisi ===============

CRYPTO_TIMEFRAME = "1d"
CRYPTO_OHLC_LIMIT = 220  # EMA için yeterli mum sayısı


def get_binance_30d_volumes(symbols: list) -> dict:
    """
    Binance public API'den 30 günlük kline çekip
    ortalama günlük hacim (USDT) hesaplar.
    
    Returns: {symbol: avg_daily_volume_usd, ...}
    """
    volumes = {}
    base_url = "https://api.binance.com/api/v3/klines"
    
    for sym in symbols:
        # BTC/USDT -> BTCUSDT formatına çevir
        if "/" in sym:
            binance_symbol = sym.replace("/", "")
        else:
            binance_symbol = sym
        
        try:
            params = {
                "symbol": binance_symbol,
                "interval": "1d",
                "limit": 30
            }
            
            r = requests.get(base_url, params=params, timeout=10)
            
            if r.status_code == 200:
                klines = r.json()
                if klines and len(klines) > 0:
                    # Her kline: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, ...]
                    # quote_asset_volume (index 7) = USDT cinsinden hacim
                    daily_volumes = [float(k[7]) for k in klines]
                    avg_volume = sum(daily_volumes) / len(daily_volumes)
                    volumes[sym] = avg_volume
            else:
                print(f"Binance hacim hatası {sym}: HTTP {r.status_code}")
                
            # Rate limit için kısa bekleme
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Binance hacim çekme hatası {sym}: {e}")
            continue
    
    return volumes


def filter_crypto_by_volume_and_blacklist(symbols: list, min_volume: float) -> list:
    """
    1. Blacklist'teki sembolleri çıkar
    2. Binance'ten 30 günlük hacim çek
    3. min_volume ($750K) üstü olanları döndür
    """
    # 1. Blacklist filtresi
    filtered = []
    for sym in symbols:
        # BTC/USDT -> BTC
        base = sym.replace("/USDT", "").replace("USDT", "").strip().upper()
        if base not in CRYPTO_BLACKLIST:
            filtered.append(sym)
        else:
            print(f"Blacklist'te: {sym}")
    
    print(f"Blacklist sonrası: {len(symbols)} -> {len(filtered)} sembol")
    
    # 2. Binance'ten hacim çek
    print(f"Binance'ten 30 günlük hacim çekiliyor ({len(filtered)} sembol)...")
    volumes = get_binance_30d_volumes(filtered)
    
    # 3. Hacim filtresi
    high_volume = []
    low_volume_count = 0
    
    for sym in filtered:
        vol = volumes.get(sym, 0)
        if vol >= min_volume:
            high_volume.append(sym)
        else:
            low_volume_count += 1
            if vol > 0:
                print(f"Düşük hacim: {sym} = ${vol:,.0f} (min: ${min_volume:,.0f})")
    
    print(f"Hacim filtresi sonrası: {len(filtered)} -> {len(high_volume)} sembol (elenen: {low_volume_count})")
    
    return high_volume


def find_mexc_symbol(binance_symbol: str, markets: dict):
    """
    Binance tarzı sembolü (BTC/USDT veya BTCUSDT) alır,
    MEXC'te olası market adını tahmin eder.
    """
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
    1. binance.txt'den sembolleri oku
    2. Blacklist ve hacim filtresi uygula (Binance verisiyle)
    3. MEXC 1D OHLCV'den EMA 13-34 bullish cross tara
    """
    result = {
        "13_34_bull": [],
        "errors": []
    }

    symbols = read_symbol_file(BINANCE_LIST_FILE)
    if not symbols:
        print(f"{BINANCE_LIST_FILE} boş veya bulunamadı.")
        return result

    print(f"\n=== Kripto Tarama Başlıyor ===")
    print(f"Toplam sembol: {len(symbols)}")
    
    # Blacklist + Hacim filtresi (Binance verisiyle)
    filtered_symbols = filter_crypto_by_volume_and_blacklist(
        symbols, 
        min_volume=CRYPTO_MIN_VOLUME_30D
    )
    
    if not filtered_symbols:
        print("Filtre sonrası sembol kalmadı!")
        return result
    
    print(f"\nMEXC'ten mum verisi çekiliyor ({len(filtered_symbols)} sembol)...")

    # MEXC bağlantısı
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

    for sym in filtered_symbols:
        raw_sym = sym.strip()
        if not raw_sym:
            continue

        mexc_symbol = find_mexc_symbol(raw_sym, markets)
        if mexc_symbol is None:
            print(f"{raw_sym}: MEXC'te market bulunamadı")
            result["errors"].append(raw_sym)
            continue

        try:
            ohlcv = exchange.fetch_ohlcv(
                mexc_symbol,
                timeframe=CRYPTO_TIMEFRAME,
                limit=CRYPTO_OHLC_LIMIT,
            )
        except Exception as e:
            print(f"MEXC veri hatası {raw_sym}: {e}")
            result["errors"].append(raw_sym)
            continue

        if not ohlcv or len(ohlcv) < 60:
            print(f"{raw_sym}: yetersiz OHLCV verisi")
            result["errors"].append(raw_sym)
            continue

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        close = df["close"].astype(float)

        display_name = raw_sym.replace("/USDT", "")

        if has_recent_bullish_cross(close, 13, 34, min_rel_gap=0.0):
            result["13_34_bull"].append(display_name)

        processed_count += 1

    print(f"\nKripto tarama tamamlandı: {processed_count} sembol işlendi")
    print(f"EMA 13-34 kesişimi: {len(result['13_34_bull'])} adet")

    return result


# =============== Formatlama ===============

def format_result_block(title: str, res: dict) -> str:
    lines = [f"📌 {title}"]

    def join_list(lst):
        return ", ".join(lst) if lst else "-"

    lines.append(f"EMA13-34 KESİŞİMİ: {join_list(res.get('13_34_bull', []))}")

    err_line = summarize_errors(res.get("errors", []))
    if err_line:
        lines.append(err_line)

    return "\n".join(lines)


# =============== Ana Akış ===============

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    header = (
        f"📊 EMA 13-34 Yükseliş Kesişim Tarama – {today_str}\n"
        f"Timeframe: 1D\n"
        f"Evren: {BIST_LABEL}, NASDAQ 100, Kripto (Binance hacim ≥$750K)\n"
        f"NOT: Sadece son 1-2 mumda oluşmuş bullish kesişimler."
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

    # --- NASDAQ 100 --- #
    nasdaq_symbols = read_symbol_file("nasdaq100.txt")
    if nasdaq_symbols:
        nasdaq_res = scan_equity_universe(nasdaq_symbols, "NASDAQ 100")
        nasdaq_text = format_result_block("🇺🇸 NASDAQ 100", nasdaq_res)
        send_telegram_message(nasdaq_text)

    # --- Kripto (Binance hacim filtresi + MEXC mum verisi) --- #
    crypto_res = scan_crypto_from_mexc_list()
    crypto_text = format_result_block("🪙 Kripto (Binance hacim ≥$750K)", crypto_res)
    send_telegram_message(crypto_text)


if __name__ == "__main__":
    main()
