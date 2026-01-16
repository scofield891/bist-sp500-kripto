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

# ---- Kripto Filtreleme Parametreleri ----
CRYPTO_TOP_K = int(os.getenv("CRYPTO_TOP_K", "320"))  # En likit kaç coin alınacak
CRYPTO_MIN_TARGET = int(os.getenv("CRYPTO_MIN_TARGET", "120"))  # Minimum hedef coin sayısı (gevşetme tetikleyici)
CRYPTO_MAX_LEVEL = int(os.getenv("CRYPTO_MAX_LEVEL", "2"))  # Maksimum gevşeme seviyesi (0-1-2, Level 3'e düşmesin)

# Tutarlılık filtreleri (başlangıç değerleri)
CRYPTO_MEDIAN_MIN = float(os.getenv("CRYPTO_MEDIAN_MIN", "600000"))  # Median >= $600K
CRYPTO_DAYS_ABOVE_FLOOR = int(os.getenv("CRYPTO_DAYS_ABOVE_FLOOR", "18"))  # 30 günün en az 18'i
CRYPTO_FLOOR_VOLUME = float(os.getenv("CRYPTO_FLOOR_VOLUME", "500000"))  # $500K floor
CRYPTO_SPIKE_RATIO_MAX = float(os.getenv("CRYPTO_SPIKE_RATIO_MAX", "8"))  # max/median <= 8

# EMA cross için minimum gap (fake cross engellemek için)
EMA_MIN_REL_GAP = float(os.getenv("EMA_MIN_REL_GAP", "0.001"))  # %0.1 (Kripto/NASDAQ)
BIST_EMA_MIN_REL_GAP = float(os.getenv("BIST_EMA_MIN_REL_GAP", "0.0005"))  # %0.05 (BIST - daha gevşek)

# BIST/NASDAQ için pencere ayarları
EQUITY_MAX_BARS_AGO = int(os.getenv("EQUITY_MAX_BARS_AGO", "2"))  # Son 2 bar
EQUITY_MAX_DAYS_AGO = int(os.getenv("EQUITY_MAX_DAYS_AGO", "5"))  # Hafta sonu kaçırmasın

def normalize_binance_base(url: str) -> str:
    """Binance base URL'ini normalize eder (sonundaki /api/v3 veya / varsa kaldırır)."""
    url = (url or "").strip().rstrip("/")
    if url.endswith("/api/v3"):
        url = url[:-7]
    return url

# Binance API base URL'leri (fallback sırasıyla denenir)
BINANCE_API_BASES = [
    normalize_binance_base(os.getenv("BINANCE_API_BASE", "https://data-api.binance.vision")),
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

# Binance API rate limit bekleme süresi (saniye)
BINANCE_RATE_LIMIT_SLEEP = float(os.getenv("BINANCE_RATE_LIMIT_SLEEP", "0.10"))

# Telegram mesaj karakter limiti (güvenli sınır)
TELEGRAM_CHAR_LIMIT = 3800

# =============== Blacklist (Stablecoin + Fan Token) ===============

CRYPTO_BLACKLIST = {
    # Stablecoinler
    "USDC", "TUSD", "FDUSD", "USDE", "USDP", "USD1", "XUSD",
    "EURI", "EUR", "BUSD", "DAI", "PAXG", "GUSD", "USDJ",
    "USDD", "USTC", "AEUR", "PYUSD", "FRAX",
    
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

def escape_html(text: str) -> str:
    """HTML özel karakterlerini escape eder."""
    return (text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))


def send_telegram_message(text: str, parse_mode: str = "HTML"):
    """
    Telegram mesajı gönderir (HTML formatında).
    4096 karakter limitini aşarsa mesajı parçalara böler.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    # Mesajı parçalara böl (4096 limit, güvenli sınır 3800)
    chunks = []
    if len(text) <= TELEGRAM_CHAR_LIMIT:
        chunks = [text]
    else:
        # Satır satır böl, her parça limite sığsın
        lines = text.split("\n")
        current_chunk = ""
        
        for line in lines:
            # Eğer bu satırı eklersek limit aşılır mı?
            if len(current_chunk) + len(line) + 1 > TELEGRAM_CHAR_LIMIT:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        # Son parçayı ekle
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    # Her parçayı gönder
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            chunk = f"<b>[{i+1}/{len(chunks)}]</b>\n{chunk}"
        
        payload = {
            "chat_id": CHAT_ID,
            "text": chunk,
            "parse_mode": parse_mode
        }
        try:
            r = requests.post(url, json=payload, timeout=20)
            if not r.ok:
                print(f"Telegram hata (parça {i+1}): {r.status_code} {r.text}")
            
            # Birden fazla parça varsa araya bekleme koy
            if len(chunks) > 1 and i < len(chunks) - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Telegram gönderim hatası (parça {i+1}): {e}")

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


def extract_base_symbol(sym: str) -> str:
    """
    Sembolden base kısmını güvenli şekilde çıkarır.
    BTC/USDT -> BTC
    BTCUSDT -> BTC
    """
    sym = sym.strip().upper()
    if "/" in sym:
        return sym.split("/")[0]
    elif sym.endswith("USDT"):
        return sym[:-4]
    else:
        return sym


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
    max_bars_ago: int = 2,
    max_days_ago: int = 5,
    min_rel_gap: float = 0.001
) -> bool:
    """
    EMA fast & slow için bullish cross noktalarını bulur.

    Şartlar:
      - Cross, son bar veya en fazla max_bars_ago bar önce olacak.
      - Cross'un tarihi bugünden en fazla max_days_ago gün önce olacak.
      - Cross barında EMA_fast - EMA_slow, fiyata oranla en az min_rel_gap olmalı.
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

    # 2) Gap kontrolü (fake cross engellemek için)
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

    # 3) Tarih bazlı kontrol (DatetimeIndex varsa)
    idx = close.index
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            last_cross_time = idx[last_cross]

            if isinstance(last_cross_time, pd.Period):
                last_cross_time = last_cross_time.to_timestamp()

            cross_ts = pd.Timestamp(last_cross_time)
            
            # BIST/yfinance tz-naive gelir, Kripto tz-aware gelir
            # Her ikisini de aynı şekilde karşılaştır
            if cross_ts.tz is None:
                # tz-naive: today de tz-naive olsun
                today_utc = pd.Timestamp.utcnow().replace(tzinfo=None).normalize()
                cross_day = cross_ts.normalize()
            else:
                # tz-aware: UTC'ye çevir
                today_utc = pd.Timestamp.now(tz="UTC").normalize()
                cross_ts = cross_ts.tz_convert("UTC")
                cross_day = cross_ts.normalize()
            
            days_diff = (today_utc - cross_day).days

            if days_diff > max_days_ago:
                return False
        except Exception as e:
            print("Tarih kontrolü hatası (has_recent_bullish_cross):", e)
            # Tarih kontrolü başarısız olursa, bar bazlı kontrole güven
            pass

    return True


def summarize_errors(errors, max_show: int = 10) -> str:
    if not errors:
        return ""
    total = len(errors)
    if total <= max_show:
        return f"<i>(Veri hatası: {', '.join(errors)})</i>"
    shown = ", ".join(errors[:max_show])
    return f"<i>(Veri hatası: {total} sembol, ilk {max_show}: {shown})</i>"


# =============== Hisse Taraması (BIST & NASDAQ, toplu yfinance) ===============

def scan_equity_universe(symbols, universe_name: str, min_gap: float = None):
    """
    yfinance ile TÜM sembolleri toplu indirip,
    EMA 13-34 için son 1 mum (max 2 mum) bullish cross arar.
    
    min_gap: EMA gap kontrolü (None ise EMA_MIN_REL_GAP kullanılır)
    """
    if min_gap is None:
        min_gap = EMA_MIN_REL_GAP
    
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

            if has_recent_bullish_cross(close, 13, 34, EQUITY_MAX_BARS_AGO, EQUITY_MAX_DAYS_AGO, min_gap):
                result["13_34_bull"].append(sym)

        except Exception as e:
            print(f"{universe_name} veri hatası {sym}: {e}")
            result["errors"].append(sym)

    return result


# =============== Kripto: Binance Hacim + Tutarlılık + MEXC Mum Verisi ===============

CRYPTO_TIMEFRAME = "1d"
CRYPTO_OHLC_LIMIT = 220  # EMA için yeterli mum sayısı


def binance_api_request(endpoint: str, params: dict = None, timeout: int = 15):
    """
    Binance API'ye istek atar.
    - 429'da aynı base üzerinde exponential backoff ile retry
    - 451/403/5xx'de sonraki base'e geç
    """
    if params is None:
        params = {}
    
    per_base_retries = 3
    backoff = 2
    
    for base in BINANCE_API_BASES:
        url = f"{base}{endpoint}"
        
        for attempt in range(per_base_retries):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception:
                        return None
                
                # 429: aynı base üzerinde bekle + retry
                if r.status_code == 429:
                    wait = backoff * (2 ** attempt)  # Exponential: 2, 4, 8
                    print(f"[Binance] 429 Rate limit ({base}) → {wait}s bekle (attempt {attempt+1}/{per_base_retries})")
                    time.sleep(wait)
                    continue
                
                # 451/403/418 veya 5xx: base değiştir
                if r.status_code in (451, 418, 403) or (500 <= r.status_code < 600):
                    print(f"[Binance] HTTP {r.status_code} ({base}) → sonraki base")
                    break
                
                # Diğer hatalar: base değiştir
                print(f"[Binance] HTTP {r.status_code} ({base}) → sonraki base")
                break
                
            except Exception as e:
                print(f"[Binance] exception ({base}) → {e}")
                break
    
    print("Tüm Binance base'leri başarısız!")
    return None


def get_binance_24h_volumes() -> dict:
    """
    Binance ticker/24hr endpoint'inden TÜM coinlerin 24h hacmini tek çağrıyla çeker.
    Returns: {symbol: quote_volume, ...}  örn: {"BTCUSDT": 1234567890.5, ...}
    """
    data = binance_api_request("/api/v3/ticker/24hr")
    if not data:
        return {}
    
    volumes = {}
    for ticker in data:
        try:
            symbol = ticker.get("symbol", "")
            if symbol.endswith("USDT"):
                quote_vol_str = ticker.get("quoteVolume", "0")
                quote_vol = float(quote_vol_str) if quote_vol_str else 0
                if quote_vol > 0:
                    volumes[symbol] = quote_vol
        except (ValueError, TypeError) as e:
            # Parse hatası olursa bu coini atla
            continue
    
    return volumes


def get_binance_klines(symbol: str, limit: int = 30) -> list:
    """
    Tek bir sembol için klines çeker (fallback destekli).
    """
    params = {"symbol": symbol, "interval": "1d", "limit": limit}
    return binance_api_request("/api/v3/klines", params)


def get_binance_30d_volume_stats(symbols: list) -> dict:
    """
    Optimize edilmiş hacim istatistikleri:
    1. Önce ticker/24hr ile tüm coinlerin anlık hacmini çek (tek istek)
    2. Sembolleri 24h hacme göre ön-filtrele
    3. Sadece filtrelenmiş coinler için 30 günlük klines çek
    
    Returns: {symbol: {avg, median, max, days_above, spike_ratio}, ...}
    """
    stats = {}
    
    # 1. Önce 24h hacimlerini tek çağrıyla al
    print("Binance'ten 24h hacim verisi çekiliyor (tek çağrı)...")
    all_24h_volumes = get_binance_24h_volumes()
    
    if not all_24h_volumes:
        print("24h hacim verisi alınamadı, doğrudan klines çekilecek...")
        # Fallback: eski yöntemle devam et
        return get_binance_30d_volume_stats_direct(symbols)
    
    print(f"24h hacim verisi alındı: {len(all_24h_volumes)} USDT çifti")
    
    # 2. Sembolleri 24h hacme göre filtrele ve sırala
    symbol_volumes = []
    for sym in symbols:
        binance_symbol = sym.replace("/", "")
        vol_24h = all_24h_volumes.get(binance_symbol, 0)
        if vol_24h > 0:
            symbol_volumes.append((sym, binance_symbol, vol_24h))
    
    # 24h hacme göre sırala (büyükten küçüğe)
    symbol_volumes.sort(key=lambda x: x[2], reverse=True)
    
    # İlk TopK + buffer kadarını al (gereksiz klines çağrısı yapmamak için)
    top_symbols = symbol_volumes[:CRYPTO_TOP_K + 150]  # Buffer artırıldı - gevşetmeye daha az ihtiyaç
    print(f"24h hacme göre ilk {len(top_symbols)} coin seçildi, klines çekiliyor...")
    
    # 3. Sadece seçilen coinler için 30 günlük klines çek
    for sym, binance_symbol, vol_24h in top_symbols:
        try:
            klines = get_binance_klines(binance_symbol, 30)
            
            if klines and len(klines) >= 7:
                # Parse guard ile daily volumes çek
                daily_volumes = []
                for k in klines:
                    try:
                        daily_volumes.append(float(k[7]))
                    except (ValueError, TypeError, IndexError):
                        continue
                
                if len(daily_volumes) < 7:
                    time.sleep(BINANCE_RATE_LIMIT_SLEEP)
                    continue
                
                avg_vol = sum(daily_volumes) / len(daily_volumes)
                sorted_vols = sorted(daily_volumes)
                n = len(sorted_vols)
                if n % 2 == 0:
                    median_vol = (sorted_vols[n // 2 - 1] + sorted_vols[n // 2]) / 2
                else:
                    median_vol = sorted_vols[n // 2]
                max_vol = max(daily_volumes)
                days_above = sum(1 for v in daily_volumes if v >= CRYPTO_FLOOR_VOLUME)
                spike_ratio = max_vol / median_vol if median_vol > 0 else 999
                
                stats[sym] = {
                    "avg": avg_vol,
                    "median": median_vol,
                    "max": max_vol,
                    "days_above": days_above,
                    "spike_ratio": spike_ratio
                }
            
            time.sleep(BINANCE_RATE_LIMIT_SLEEP)
            
        except Exception as e:
            print(f"Klines hatası {sym}: {e}")
            continue
    
    return stats


def get_binance_30d_volume_stats_direct(symbols: list) -> dict:
    """
    Fallback: Doğrudan her sembol için klines çeker (eski yöntem).
    ticker/24hr çalışmazsa kullanılır.
    """
    stats = {}
    
    for sym in symbols:
        binance_symbol = sym.replace("/", "")
        
        try:
            klines = get_binance_klines(binance_symbol, 30)
            
            if klines and len(klines) >= 7:
                # Parse guard ile daily volumes çek
                daily_volumes = []
                for k in klines:
                    try:
                        daily_volumes.append(float(k[7]))
                    except (ValueError, TypeError, IndexError):
                        continue
                
                if len(daily_volumes) < 7:
                    time.sleep(BINANCE_RATE_LIMIT_SLEEP)
                    continue
                
                avg_vol = sum(daily_volumes) / len(daily_volumes)
                sorted_vols = sorted(daily_volumes)
                n = len(sorted_vols)
                if n % 2 == 0:
                    median_vol = (sorted_vols[n // 2 - 1] + sorted_vols[n // 2]) / 2
                else:
                    median_vol = sorted_vols[n // 2]
                max_vol = max(daily_volumes)
                days_above = sum(1 for v in daily_volumes if v >= CRYPTO_FLOOR_VOLUME)
                spike_ratio = max_vol / median_vol if median_vol > 0 else 999
                
                stats[sym] = {
                    "avg": avg_vol,
                    "median": median_vol,
                    "max": max_vol,
                    "days_above": days_above,
                    "spike_ratio": spike_ratio
                }
                
            time.sleep(BINANCE_RATE_LIMIT_SLEEP)
            
        except Exception as e:
            print(f"Binance hacim çekme hatası {sym}: {e}")
            continue
    
    return stats


def apply_consistency_filter(
    top_k_symbols: list,
    median_min: float,
    days_above_min: int,
    spike_max: float
) -> tuple:
    """
    Tutarlılık filtresini uygular ve geçen sembolleri döndürür.
    """
    final_list = []
    filtered_out = {"median": 0, "days_above": 0, "spike_ratio": 0}
    
    for sym, stat in top_k_symbols:
        # Median kontrolü
        if stat["median"] < median_min:
            filtered_out["median"] += 1
            continue
        
        # Days above floor kontrolü
        if stat["days_above"] < days_above_min:
            filtered_out["days_above"] += 1
            continue
        
        # Spike ratio kontrolü
        if stat["spike_ratio"] > spike_max:
            filtered_out["spike_ratio"] += 1
            continue
        
        final_list.append(sym)
    
    return final_list, filtered_out


def filter_crypto_symbols(symbols: list) -> tuple:
    """
    Kripto filtreleme pipeline:
    1. Blacklist'teki sembolleri çıkar
    2. Binance'ten 30 günlük hacim istatistikleri çek
    3. TopK en likit olanı seç
    4. Tutarlılık filtresi uygula (median, days_above, spike_ratio)
    5. 200+ garanti için otomatik gevşetme
    
    Returns: (final_list, used_level, final_count)
    """
    # 1. Blacklist filtresi (güvenli base çıkarma)
    after_blacklist = []
    for sym in symbols:
        base = extract_base_symbol(sym)
        if base not in CRYPTO_BLACKLIST:
            after_blacklist.append(sym)
        else:
            print(f"Blacklist'te: {sym}")
    
    print(f"Blacklist sonrası: {len(symbols)} -> {len(after_blacklist)} sembol")
    
    if not after_blacklist:
        return [], 0, 0
    
    # 2. Binance'ten hacim istatistikleri çek (optimize edilmiş)
    print(f"Binance'ten hacim istatistikleri çekiliyor ({len(after_blacklist)} sembol)...")
    volume_stats = get_binance_30d_volume_stats(after_blacklist)
    
    if not volume_stats:
        print("Hacim verisi alınamadı!")
        return [], 0, 0
    
    print(f"Hacim verisi alınan: {len(volume_stats)} sembol")
    
    # 3. Median hacme göre sırala ve TopK seç
    sorted_by_median = sorted(
        volume_stats.items(),
        key=lambda x: x[1]["median"],
        reverse=True
    )
    
    top_k_symbols = sorted_by_median[:CRYPTO_TOP_K]
    print(f"TopK ({CRYPTO_TOP_K}) seçildi: {len(top_k_symbols)} sembol")
    
    # 4. Tutarlılık filtresi + 200+ garanti için otomatik gevşetme
    # Gevşetme seviyeleri (sırasıyla denenecek)
    relaxation_levels = [
        # Level 0: Orijinal değerler
        {"median_min": CRYPTO_MEDIAN_MIN, "days_above_min": CRYPTO_DAYS_ABOVE_FLOOR, "spike_max": CRYPTO_SPIKE_RATIO_MAX, "label": "Sıkı"},
        # Level 1: days_above gevşet
        {"median_min": CRYPTO_MEDIAN_MIN, "days_above_min": 16, "spike_max": CRYPTO_SPIKE_RATIO_MAX, "label": "Normal"},
        # Level 2: median gevşet
        {"median_min": 500000, "days_above_min": 16, "spike_max": CRYPTO_SPIKE_RATIO_MAX, "label": "Gevşek"},
        # Level 3: spike gevşet
        {"median_min": 500000, "days_above_min": 16, "spike_max": 10, "label": "Çok Gevşek"},
        # Level 4: daha da gevşet
        {"median_min": 400000, "days_above_min": 14, "spike_max": 12, "label": "Minimum"},
    ]
    
    final_list = []
    used_level = 0
    filtered_out = {}
    
    for level_idx, params in enumerate(relaxation_levels):
        # Maksimum gevşeme seviyesini aşma
        if level_idx > CRYPTO_MAX_LEVEL:
            print(f"Maksimum gevşeme seviyesine ({CRYPTO_MAX_LEVEL}) ulaşıldı, daha fazla gevşetme yapılmayacak.")
            break
        
        final_list, filtered_out = apply_consistency_filter(
            top_k_symbols,
            params["median_min"],
            params["days_above_min"],
            params["spike_max"]
        )
        
        if len(final_list) >= CRYPTO_MIN_TARGET:
            used_level = level_idx
            break
        else:
            print(f"Level {level_idx} ({params['label']}): {len(final_list)} sembol (hedef: {CRYPTO_MIN_TARGET}), gevşetiliyor...")
            used_level = level_idx
    
    level_label = relaxation_levels[used_level]["label"]
    
    print(f"\n=== Tutarlılık Filtresi Sonucu ===")
    print(f"Kullanılan filtre seviyesi: Level {used_level} ({level_label})")
    print(f"TopK'dan geçen: {len(final_list)} sembol")
    print(f"Elenenler -> Median düşük: {filtered_out['median']}, "
          f"Tutarsız: {filtered_out['days_above']}, "
          f"Spike: {filtered_out['spike_ratio']}")
    
    if len(final_list) < CRYPTO_MIN_TARGET:
        print(f"UYARI: Hedef ({CRYPTO_MIN_TARGET}) karşılanamadı, {len(final_list)} sembolle devam ediliyor.")
    
    return final_list, used_level, len(final_list)


def find_mexc_symbol(binance_symbol: str, markets: dict):
    """
    Binance tarzı sembolü (BTC/USDT veya BTCUSDT) alır,
    MEXC'te olası market adını tahmin eder.
    """
    s = binance_symbol.strip().upper()
    if not s:
        return None

    base = extract_base_symbol(s)
    quote = "USDT"

    candidates = [
        f"{base}/{quote}",
        f"{base}/{quote}:USDT",
    ]

    for c in candidates:
        if c in markets:
            return c

    return None


def remove_incomplete_candle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bugünkü tamamlanmamış mumu kaldırır.
    1D timeframe'de son bar bugünse atılır.
    Tüm karşılaştırmalar UTC tz-aware yapılır.
    """
    if df.empty:
        return df
    
    last_ts = df["timestamp"].iloc[-1]
    
    # ccxt: ms int -> UTC tz-aware midnight
    last_date = pd.Timestamp(int(last_ts), unit="ms", tz="UTC").normalize()
    
    # UTC tz-aware today midnight
    today = pd.Timestamp.now(tz="UTC").normalize()
    
    if last_date >= today:
        return df.iloc[:-1]
    
    return df


def scan_crypto_from_mexc_list() -> tuple:
    """
    1. binance.txt'den sembolleri oku
    2. Blacklist + TopK + Tutarlılık filtresi uygula
    3. MEXC 1D OHLCV'den EMA 13-34 bullish cross tara
    4. Bugünkü yarım mumu atarak fake sinyalleri azalt
    5. Timestamp index kullanarak tarih kontrolü çalışsın
    
    Returns: (result_dict, filter_level, scanned_count)
    """
    result = {
        "13_34_bull": [],
        "errors": []
    }

    symbols = read_symbol_file(BINANCE_LIST_FILE)
    if not symbols:
        print(f"{BINANCE_LIST_FILE} boş veya bulunamadı.")
        return result, 0, 0

    print(f"\n{'='*50}")
    print(f"=== Kripto Tarama Başlıyor ===")
    print(f"{'='*50}")
    print(f"Toplam sembol: {len(symbols)}")
    
    # Blacklist + TopK + Tutarlılık filtresi
    filtered_symbols, filter_level, filter_count = filter_crypto_symbols(symbols)
    
    if not filtered_symbols:
        print("Filtre sonrası sembol kalmadı!")
        return result, filter_level, 0
    
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
        return result, filter_level, 0

    processed_count = 0
    mexc_not_found = 0

    for sym in filtered_symbols:
        raw_sym = sym.strip()
        if not raw_sym:
            continue

        mexc_symbol = find_mexc_symbol(raw_sym, markets)
        if mexc_symbol is None:
            print(f"{raw_sym}: MEXC'te market bulunamadı")
            mexc_not_found += 1
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
        
        # KRITIK: Bugünkü tamamlanmamış mumu at
        df = remove_incomplete_candle(df)
        
        if len(df) < 60:
            print(f"{raw_sym}: yarım mum atıldıktan sonra yetersiz veri")
            result["errors"].append(raw_sym)
            continue
        
        # Timestamp'i DatetimeIndex'e çevir (tarih kontrolü çalışsın - UTC tz-aware)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        close = pd.Series(df["close"].astype(float).values, index=df["dt"])

        display_name = extract_base_symbol(raw_sym)

        if has_recent_bullish_cross(close, 13, 34, EQUITY_MAX_BARS_AGO, EQUITY_MAX_DAYS_AGO, EMA_MIN_REL_GAP):
            result["13_34_bull"].append(display_name)

        processed_count += 1

    print(f"\n{'='*50}")
    print(f"=== Kripto Tarama Tamamlandı ===")
    print(f"{'='*50}")
    print(f"Filtre sonrası: {len(filtered_symbols)} sembol")
    print(f"MEXC'te bulunamayan: {mexc_not_found}")
    print(f"Başarıyla işlenen: {processed_count}")
    print(f"EMA 13-34 kesişimi: {len(result['13_34_bull'])} adet")

    return result, filter_level, processed_count


# =============== Formatlama (HTML) ===============

def format_result_block(title: str, res: dict) -> str:
    """
    Sonuçları HTML formatında formatlar.
    Coin listesi <code> bloğunda monospace gösterilir.
    Uzun listeler 25 coin/satır olarak bölünür.
    """
    lines = [f"<b>📌 {escape_html(title)}</b>"]

    def format_coin_list(lst, per_line=25):
        if not lst:
            return "<i>-</i>"
        # Coin listesini satırlara böl (Telegram 4096 limit için)
        result_lines = []
        for i in range(0, len(lst), per_line):
            result_lines.append(", ".join(lst[i:i+per_line]))
        coins_str = "\n".join(result_lines)
        return f"<code>{escape_html(coins_str)}</code>"

    bull_list = res.get('13_34_bull', [])
    count_str = f" ({len(bull_list)} adet)" if bull_list else ""
    lines.append(f"<b>EMA13-34 KESİŞİMİ{count_str}:</b>")
    lines.append(format_coin_list(bull_list))

    err_line = summarize_errors(res.get("errors", []))
    if err_line:
        lines.append(err_line)

    return "\n".join(lines)


def get_filter_level_label(level: int) -> str:
    """Filtre seviyesinin Türkçe etiketini döndürür."""
    labels = {
        0: "Sıkı 🔒",
        1: "Normal ✅",
        2: "Gevşek ⚡",
        3: "Çok Gevşek ⚠️",
        4: "Minimum 🔓"
    }
    return labels.get(level, f"Level {level}")


# =============== Ana Akış ===============

def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # --- BIST (havuzdan likiditeye göre TOP N) --- #
    bist_all = read_symbol_file(BIST_ALL_FILE)
    bist_text = None
    bist_symbols = []

    if bist_all:
        bist_symbols = select_most_liquid_bist_symbols(
            bist_all,
            max_count=BIST_MAX_COUNT,
            universe_name="BIST Likit"
        )

        if bist_symbols:
            bist_res = scan_equity_universe(bist_symbols, "BIST Likit", min_gap=BIST_EMA_MIN_REL_GAP)
            bist_label_full = f"🇹🇷 {BIST_LABEL} ({len(bist_symbols)} hisse)"
            bist_text = format_result_block(bist_label_full, bist_res)
    else:
        print(f"{BIST_ALL_FILE} bulunamadı, BIST taraması yapılmayacak.")

    # --- NASDAQ 100 --- #
    nasdaq_symbols = read_symbol_file("nasdaq100.txt")
    nasdaq_text = None
    
    if nasdaq_symbols:
        nasdaq_res = scan_equity_universe(nasdaq_symbols, "NASDAQ 100", min_gap=EMA_MIN_REL_GAP)
        nasdaq_text = format_result_block("🇺🇸 NASDAQ 100", nasdaq_res)

    # --- Kripto (TopK + Tutarlılık filtresi + MEXC mum verisi) --- #
    crypto_res, crypto_filter_level, crypto_scanned = scan_crypto_from_mexc_list()
    filter_label = get_filter_level_label(crypto_filter_level)
    crypto_text = format_result_block(f"🪙 Kripto ({crypto_scanned} coin tarandı)", crypto_res)

    # --- Telegram'a gönder (header + sonuçlar) --- #
    header = (
        f"<b>📊 EMA 13-34 Yükseliş Kesişim Tarama</b>\n"
        f"<b>Tarih:</b> {today_str}\n"
        f"<b>Timeframe:</b> 1D\n"
        f"<b>Evren:</b> {BIST_LABEL}, NASDAQ 100, Kripto Top {CRYPTO_TOP_K}\n"
        f"<b>Kripto Filtre:</b> {filter_label} (Level {crypto_filter_level})\n"
        f"<i>NOT: Sadece son 1-2 mumda oluşmuş bullish kesişimler.</i>"
    )
    send_telegram_message(header)

    if bist_text:
        send_telegram_message(bist_text)
    
    if nasdaq_text:
        send_telegram_message(nasdaq_text)
    
    send_telegram_message(crypto_text)


if __name__ == "__main__":
    main()
