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

# Kripto tarafı: Midas sembol listesi dosyası
CRYPTO_LIST_FILE = os.getenv("CRYPTO_LIST_FILE", "midas.txt")

# ---- Kripto Filtreleme Parametreleri ----
CRYPTO_MIN_TARGET = int(os.getenv("CRYPTO_MIN_TARGET", "80"))  # Minimum hedef coin sayısı
CRYPTO_MAX_COINS = int(os.getenv("CRYPTO_MAX_COINS", "100"))  # Maximum taranacak coin sayısı
CRYPTO_MAX_LEVEL = int(os.getenv("CRYPTO_MAX_LEVEL", "2"))  # Maksimum gevşeme seviyesi

# CoinGecko hacim filtreleri
CRYPTO_MIN_24H_VOLUME = float(os.getenv("CRYPTO_MIN_24H_VOLUME", "5000000"))  # Min 24h hacim $5M (Level 0)
CRYPTO_MIN_MCAP = float(os.getenv("CRYPTO_MIN_MCAP", "50000000"))  # Min market cap $50M (Level 0)

# EMA cross için minimum gap (fake cross engellemek için)
EMA_MIN_REL_GAP = float(os.getenv("EMA_MIN_REL_GAP", "0.001"))  # %0.1 (Kripto/NASDAQ)
BIST_EMA_MIN_REL_GAP = float(os.getenv("BIST_EMA_MIN_REL_GAP", "0.0005"))  # %0.05 (BIST - daha gevşek)

# BIST/NASDAQ için pencere ayarları
EQUITY_MAX_BARS_AGO = int(os.getenv("EQUITY_MAX_BARS_AGO", "2"))  # Son 2 bar
EQUITY_MAX_DAYS_AGO = int(os.getenv("EQUITY_MAX_DAYS_AGO", "5"))  # Hafta sonu kaçırmasın

# CoinGecko API
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Telegram mesaj karakter limiti (güvenli sınır)
TELEGRAM_CHAR_LIMIT = 3800

# ---- TCE Super App API ----
TCE_API_URL = os.getenv("TCE_API_URL", "http://144.24.164.111:8100")

# =============== Blacklist (Stablecoin + Fan Token) ===============

CRYPTO_BLACKLIST = {
    # Stablecoinler
    "USDC", "TUSD", "FDUSD", "USDE", "USDP", "USD1", "XUSD",
    "EURI", "EUR", "BUSD", "DAI", "PAXG", "GUSD", "USDJ",
    "USDD", "USTC", "AEUR", "PYUSD", "FRAX", "USDT", "XAUT",
    
    # Fan tokenler
    "BAR", "PSG", "SANTOS", "LAZIO", "PORTO", "ACM", "ASR",
    "CITY", "ALPINE", "OG", "JUV", "ATM", "INTER", "AFC",
    "NAV", "SPURS",
    
    # Wrapped tokenler
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
    
    chunks = []
    if len(text) <= TELEGRAM_CHAR_LIMIT:
        chunks = [text]
    else:
        lines = text.split("\n")
        current_chunk = ""
        
        for line in lines:
            if len(current_chunk) + len(line) + 1 > TELEGRAM_CHAR_LIMIT:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            chunk = f"<b>[{i+1}/{len(chunks)}]</b>\n{chunk}"
        
        payload = {
            "chat_id": CHAT_ID,
            "text": chunk,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            r = requests.post(url, json=payload, timeout=20)
            if not r.ok:
                print(f"Telegram hata (parça {i+1}): {r.status_code} {r.text}")
            
            if len(chunks) > 1 and i < len(chunks) - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Telegram gönderim hatası (parça {i+1}): {e}")

# =============== Ortak Yardımcılar ===============

def read_symbol_file(path: str):
    """
    Sembol listesi dosyasını okur.
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
    Sembolden base kısmını çıkarır.
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
    BIST sembolleri arasından en likit olanları seçer.
    """
    if not symbols:
        return []

    try:
        data = yf.download(
            symbols,
            period=f"{lookback_days}d",
            interval=TIMEFRAME_DAYS,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"{universe_name} likidite indirme hatası:", e)
        return symbols

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
        print(f"{universe_name} için likidite listesi boş, fallback.")
        return symbols

    liquidity_list.sort(key=lambda x: x[1], reverse=True)
    top_syms = [sym for sym, _ in liquidity_list[:max_count]]

    print(f"{universe_name}: {len(symbols)} sembolden {len(top_syms)} seçildi.")
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
    EMA bullish cross kontrolü.
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
    bars_ago = last_idx - last_cross
    if bars_ago > max_bars_ago:
        return False

    # 2) Gap kontrolü — son bardaki gap'e bakıyoruz (cross barında gap çok küçük olabilir)
    if min_rel_gap > 0:
        try:
            gap = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
            price = float(close.iloc[-1])
            if price <= 0 or gap <= 0:
                return False
            rel_gap = gap / price
            if rel_gap < min_rel_gap:
                return False
        except Exception:
            return False

    # 3) Sanity check: Fiyat EMA'dan çok uzaksa veri adjust sorunu var, skip et
    try:
        price = float(close.iloc[-1])
        ema_f = float(ema_fast.iloc[-1])
        if price > 0 and ema_f > 0:
            price_vs_ema = abs(price - ema_f) / ema_f
            if price_vs_ema > 0.15:
                return False
    except Exception:
        pass

    # 4) Tarih bazlı kontrol
    idx = close.index
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            last_cross_time = idx[last_cross]

            if isinstance(last_cross_time, pd.Period):
                last_cross_time = last_cross_time.to_timestamp()

            cross_ts = pd.Timestamp(last_cross_time)
            
            if cross_ts.tz is None:
                today_utc = pd.Timestamp.utcnow().replace(tzinfo=None).normalize()
                cross_day = cross_ts.normalize()
            else:
                today_utc = pd.Timestamp.now(tz="UTC").normalize()
                cross_ts = cross_ts.tz_convert("UTC")
                cross_day = cross_ts.normalize()
            
            days_diff = (today_utc - cross_day).days

            if days_diff > max_days_ago:
                return False
        except Exception:
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


# =============== Hisse Taraması (BIST & NASDAQ) ===============

def remove_incomplete_candle_equity(close: pd.Series) -> pd.Series:
    """
    FIX: yfinance bugünün yarım mumunu da veriyor.
    Piyasa açıkken çalışırsa sahte EMA kesişimine yol açar.
    Son bar bugüne aitse onu kaldırıyoruz.
    """
    if close.empty:
        return close
    try:
        today = pd.Timestamp.now(tz="UTC").normalize()
        if close.index.tz is None:
            today_naive = today.tz_localize(None)
            close = close[close.index.normalize() < today_naive]
        else:
            close = close[close.index.tz_convert("UTC").normalize() < today]
    except Exception:
        pass
    return close


def scan_equity_universe(symbols, universe_name: str, min_gap: float = None):
    """
    yfinance ile hisse taraması yapar.
    """
    if min_gap is None:
        min_gap = EMA_MIN_REL_GAP
    
    result = {
        "13_34_bull": [],
        "errors": []
    }

    if not symbols:
        return result

    download_symbols = symbols if len(symbols) > 1 else symbols * 2

    try:
        data = yf.download(
            download_symbols,
            period="400d",
            interval=TIMEFRAME_DAYS,
            group_by="ticker",
            auto_adjust=True,
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

            close = remove_incomplete_candle_equity(close)

            if close.empty or len(close) < 40:
                result["errors"].append(sym)
                continue

            if has_recent_bullish_cross(close, 13, 34, EQUITY_MAX_BARS_AGO, EQUITY_MAX_DAYS_AGO, min_gap):
                result["13_34_bull"].append(sym)

        except Exception as e:
            print(f"{universe_name} veri hatası {sym}: {e}")
            result["errors"].append(sym)

    return result


# =============== CoinGecko Hacim Verisi ===============

def get_coingecko_market_data() -> dict:
    """
    CoinGecko'dan top coinlerin market verisini çeker.
    Tek çağrıda 250 coin, 2 çağrıda 500 coin.
    """
    market_data = {}
    
    for page in [1, 2]:
        try:
            url = f"{COINGECKO_API_URL}/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false"
            }
            
            r = requests.get(url, params=params, timeout=30)
            
            if r.status_code == 200:
                data = r.json()
                for coin in data:
                    symbol = coin.get("symbol", "").upper()
                    if symbol:
                        new_mcap = coin.get("market_cap", 0) or 0
                        if symbol in market_data:
                            existing_mcap = market_data[symbol]["market_cap"]
                            if new_mcap <= existing_mcap:
                                continue
                        market_data[symbol] = {
                            "volume_24h": coin.get("total_volume", 0) or 0,
                            "market_cap": new_mcap,
                            "price": coin.get("current_price", 0) or 0,
                            "name": coin.get("name", ""),
                            "id": coin.get("id", "")
                        }
                print(f"CoinGecko sayfa {page}: {len(data)} coin alındı")
            else:
                print(f"CoinGecko hata (sayfa {page}): HTTP {r.status_code}")
            
            if page < 2:
                time.sleep(1.5)
                
        except Exception as e:
            print(f"CoinGecko exception (sayfa {page}): {e}")
    
    print(f"CoinGecko toplam: {len(market_data)} coin")
    return market_data


def filter_crypto_by_coingecko(symbols: list, market_data: dict) -> tuple:
    """
    CoinGecko hacim verisiyle kripto filtreleme.
    """
    after_blacklist = []
    for sym in symbols:
        base = extract_base_symbol(sym)
        if base not in CRYPTO_BLACKLIST:
            after_blacklist.append((sym, base))
    
    print(f"Blacklist sonrası: {len(symbols)} -> {len(after_blacklist)} sembol")
    
    if not after_blacklist:
        return [], 0, {}
    
    matched = []
    not_found = []
    
    for sym, base in after_blacklist:
        if base in market_data:
            data = market_data[base]
            matched.append({
                "symbol": sym,
                "base": base,
                "volume_24h": data["volume_24h"],
                "market_cap": data["market_cap"]
            })
        else:
            not_found.append(base)
    
    print(f"CoinGecko eşleşen: {len(matched)}, bulunamayan: {len(not_found)}")
    if not_found[:10]:
        print(f"Bulunamayan örnekler: {not_found[:10]}")
    
    if not matched:
        return [], 0, {"matched": 0, "not_found": len(not_found)}
    
    matched.sort(key=lambda x: x["volume_24h"], reverse=True)
    
    relaxation_levels = [
        {"min_volume": 5000000, "min_mcap": 50000000, "label": "Sıkı"},
        {"min_volume": 2000000, "min_mcap": 20000000, "label": "Normal"},
        {"min_volume": 1000000, "min_mcap": 10000000, "label": "Gevşek"},
    ]
    
    final_list = []
    used_level = 0
    
    for level_idx, params in enumerate(relaxation_levels):
        if level_idx > CRYPTO_MAX_LEVEL:
            print(f"Maksimum gevşeme seviyesine ({CRYPTO_MAX_LEVEL}) ulaşıldı.")
            break
        
        filtered = []
        for coin in matched:
            if coin["volume_24h"] >= params["min_volume"] and coin["market_cap"] >= params["min_mcap"]:
                filtered.append(coin["symbol"])
        
        if len(filtered) >= CRYPTO_MIN_TARGET:
            final_list = filtered
            used_level = level_idx
            break
        else:
            print(f"Level {level_idx} ({params['label']}): {len(filtered)} sembol (hedef: {CRYPTO_MIN_TARGET}), gevşetiliyor...")
            final_list = filtered
            used_level = level_idx
    
    level_label = relaxation_levels[used_level]["label"]
    print(f"Kripto filtre: Level {used_level} ({level_label}), geçen: {len(final_list)} sembol")
    
    if len(final_list) > CRYPTO_MAX_COINS:
        print(f"Max {CRYPTO_MAX_COINS} coin'e kırpılıyor ({len(final_list)} -> {CRYPTO_MAX_COINS})")
        final_list = final_list[:CRYPTO_MAX_COINS]
    
    stats = {
        "matched": len(matched),
        "not_found": len(not_found),
        "filtered": len(final_list)
    }
    
    return final_list, used_level, stats


# =============== Kripto Tarama (MEXC) ===============

CRYPTO_TIMEFRAME = "1d"
CRYPTO_OHLC_LIMIT = 220


def find_mexc_symbol(base_symbol: str, markets: dict):
    base = extract_base_symbol(base_symbol).upper()
    if not base:
        return None
    candidates = [f"{base}/USDT", f"{base}/USDT:USDT"]
    for c in candidates:
        if c in markets:
            return c
    return None


def remove_incomplete_candle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    last_ts = df["timestamp"].iloc[-1]
    last_date = pd.Timestamp(int(last_ts), unit="ms", tz="UTC").normalize()
    today = pd.Timestamp.now(tz="UTC").normalize()
    if last_date >= today:
        return df.iloc[:-1]
    return df


def scan_crypto_from_list() -> tuple:
    """
    Kripto tarama: midas.txt -> CoinGecko filtre -> MEXC mum -> EMA 13-34
    """
    result = {
        "13_34_bull": [],
        "errors": []
    }

    symbols = read_symbol_file(CRYPTO_LIST_FILE)
    if not symbols:
        print(f"{CRYPTO_LIST_FILE} boş veya bulunamadı.")
        return result, 0, 0

    print(f"\n{'='*50}")
    print(f"=== Kripto Tarama Başlıyor ===")
    print(f"{'='*50}")
    print(f"Toplam sembol: {len(symbols)}")
    
    print("\nCoinGecko'dan hacim verisi çekiliyor...")
    market_data = get_coingecko_market_data()
    
    if not market_data:
        print("CoinGecko verisi alınamadı!")
        return result, 0, 0
    
    filtered_symbols, filter_level, stats = filter_crypto_by_coingecko(symbols, market_data)
    
    if not filtered_symbols:
        print("Filtre sonrası sembol kalmadı!")
        return result, filter_level, 0
    
    print(f"\nMEXC'ten mum verisi çekiliyor ({len(filtered_symbols)} sembol)...")

    try:
        exchange = ccxt.mexc({"enableRateLimit": True})
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

        base = extract_base_symbol(raw_sym)
        mexc_symbol = find_mexc_symbol(raw_sym, markets)
        
        if mexc_symbol is None:
            mexc_not_found += 1
            result["errors"].append(base)
            continue

        try:
            ohlcv = exchange.fetch_ohlcv(mexc_symbol, timeframe=CRYPTO_TIMEFRAME, limit=CRYPTO_OHLC_LIMIT)
        except Exception as e:
            print(f"MEXC veri hatası {base}: {e}")
            result["errors"].append(base)
            continue

        if not ohlcv or len(ohlcv) < 60:
            result["errors"].append(base)
            continue

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = remove_incomplete_candle(df)
        
        if len(df) < 60:
            result["errors"].append(base)
            continue
        
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        close = pd.Series(df["close"].astype(float).values, index=df["dt"])

        if has_recent_bullish_cross(close, 13, 34, EQUITY_MAX_BARS_AGO, EQUITY_MAX_DAYS_AGO, EMA_MIN_REL_GAP):
            result["13_34_bull"].append(base)

        processed_count += 1

    print(f"\n{'='*50}")
    print(f"=== Kripto Tarama Tamamlandı ===")
    print(f"{'='*50}")
    print(f"Filtre sonrası: {len(filtered_symbols)} sembol")
    print(f"MEXC'te bulunamayan: {mexc_not_found}")
    print(f"Başarıyla işlenen: {processed_count}")
    print(f"EMA 13-34 kesişimi: {len(result['13_34_bull'])} adet")

    return result, filter_level, processed_count


# =============== TCE Piyasa Filtresi ===============

def fetch_tce_scores(retries: int = 3, delay: int = 5) -> dict:
    """
    TCE Super App API'den skor verisi çeker.
    Strateji: Önce cache'li endpoint (hızlı), yoksa refresh (yavaş).
    Retry: 3 deneme, arada 5 saniye bekle.
    """
    # 1) Önce cache'li endpoint dene (hızlı)
    try:
        print("TCE API: cache'li skor deneniyor...")
        r = requests.get(f"{TCE_API_URL}/api/scores", timeout=30)
        if r.status_code == 200:
            data = r.json()
            if "error" not in data and data.get("crypto", {}).get("score", -1) >= 0:
                print("TCE API: cache'li skor alindi.")
                return data
    except Exception as e:
        print(f"TCE API cache hatasi: {e}")

    # 2) Cache bossa/hataysa → refresh (tam hesaplama)
    for attempt in range(1, retries + 1):
        try:
            print(f"TCE API refresh deneme {attempt}/{retries}...")
            r = requests.get(f"{TCE_API_URL}/api/scores/refresh", timeout=90)
            if r.status_code == 200:
                data = r.json()
                if "error" in data:
                    print(f"TCE API hata response: {data['error']}")
                    if attempt < retries:
                        time.sleep(delay)
                        continue
                    return None
                return data
            else:
                print(f"TCE API hata: HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            print(f"TCE API timeout (deneme {attempt})")
        except Exception as e:
            print(f"TCE API baglanti hatasi (deneme {attempt}): {e}")

        if attempt < retries:
            print(f"{delay}s bekleniyor...")
            time.sleep(delay)

    print("TCE API: Tum denemeler basarisiz.")
    return None


def format_tce_message(data: dict) -> str:
    """
    TCE v2.5 skorlarını Telegram mesajı olarak formatlar.
    Format: Skor | Rejim | Boyut + Stres + Flow + Teknik
    """
    lines = ["📊 TCE Piyasa Filtresi v2.5", ""]

    # Rejim emoji mapping
    REJIM_EMOJI = {
        "KRIZ":              "⛔",
        "ZAYIF PIYASA":      "⚠️",
        "KARARSIZ":          "🔸",
        "TOPARLANAN PIYASA": "📈",
        "GUCLU PIYASA":      "🚀",
        "UNKNOWN":           "❓",
    }

    # Stres emoji
    STRES_EMOJI = {
        "NORMAL":  "🟢",
        "DIKKAT":  "🟡",
        "YUKSEK":  "🟠",
        "ASIRI":   "🔴",
    }

    # Teknik faz → kısa etiket
    PHASE_MAP = {
        "GUCLU_TREND": "Güçlü Trend",
        "YORGUN_TREND": "Yorgun Trend",
        "GECIS": "Geçiş",
        "DIP_OLUSUMU": "Dip Oluşumu",
        "RISKLI": "Riskli",
    }

    markets = [
        ("₿", "Kripto", data.get("crypto", {})),
        ("🇹🇷", "BIST", data.get("bist", {})),
        ("🇺🇸", "S&P 500", data.get("sp500", {})),
    ]

    for icon, name, md in markets:
        if not md:
            lines.append(f"{icon} {name}: Veri yok")
            lines.append("")
            continue

        score = md.get("score", 0)
        regime_data = md.get("regime", {})
        action_data = md.get("action", {})
        stress_data = md.get("stress", {})
        flow_data = md.get("flow_quality", {})
        phase = md.get("phase", None)

        regime_name = regime_data.get("regime", "UNKNOWN")
        regime_emoji = REJIM_EMOJI.get(regime_name, "❓")
        boyut = action_data.get("size", "?")

        # Ana satır: Piyasa | Skor | Rejim | Boyut
        lines.append(f"{icon} {name}: {score} | {regime_emoji} {regime_name} | {boyut}")

        # Detay satırı: Stres + Flow (varsa) + Teknik
        details = []

        # Stres
        stres_label = stress_data.get("label", "")
        if stres_label:
            stres_e = STRES_EMOJI.get(stres_label, "")
            details.append(f"Stres: {stres_e}{stres_label}")

        # Flow Quality (sadece kripto)
        if isinstance(flow_data, dict) and flow_data.get("label"):
            details.append(f"Akış: {flow_data['label']}")

        # Teknik faz
        if phase and phase not in ("UNKNOWN",):
            phase_text = PHASE_MAP.get(phase, phase)
            details.append(f"Teknik: {phase_text}")

        if details:
            lines.append(" | ".join(details))

        lines.append("")

    # Confidence
    conf = data.get("confidence", {})
    conf_label = conf.get("label", "?") if isinstance(conf, dict) else "?"
    locked = data.get("score_locked", False)

    saat = datetime.utcnow().strftime("%H:%M")
    footer = f"Güven: {conf_label.capitalize()} | Saat: {saat} UTC"
    if locked:
        footer += " | ⚠️ Skor Kilitli"
    lines.append(footer)

    return "\n".join(lines)


# =============== Formatlama (HTML) ===============

def format_result_block(title: str, res: dict) -> str:
    """
    Sonuçları HTML formatında formatlar.
    """
    lines = [f"<b>📌 {escape_html(title)}</b>"]

    def format_coin_list(lst, per_line=25):
        if not lst:
            return "<i>-</i>"
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
    labels = {
        0: "Sıkı 🔒",
        1: "Normal ✅",
        2: "Gevşek ⚡",
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

    # --- Kripto (CoinGecko hacim + MEXC mum) --- #
    crypto_res, crypto_filter_level, crypto_scanned = scan_crypto_from_list()
    filter_label = get_filter_level_label(crypto_filter_level)
    crypto_text = format_result_block(f"🪙 Kripto ({crypto_scanned} coin tarandı)", crypto_res)

    # --- TCE Piyasa Filtresi --- #
    print("\nTCE skorları çekiliyor...")
    tce_data = fetch_tce_scores()
    tce_text = None
    if tce_data:
        tce_text = format_tce_message(tce_data)
        print("TCE skorları alındı.")
    else:
        tce_text = "📊 TCE Piyasa Filtresi\n\n⚠️ API'ye ulaşılamadı. Skorlar dashboard'dan kontrol edilebilir.\nhttp://144.24.164.111:8101"
        print("TCE skorları alınamadı, uyarı mesajı gönderilecek.")

    # --- Telegram'a gönder --- #
    header = (
        f"<b>📊 EMA 13-34 Yükseliş Kesişim Tarama</b>\n"
        f"<b>Tarih:</b> {today_str}\n"
        f"<b>Timeframe:</b> 1D\n"
        f"<b>Evren:</b> {BIST_LABEL}, NASDAQ 100, Midas Kripto\n"
        f"<b>Kripto Filtre:</b> {filter_label} (Level {crypto_filter_level})\n"
        f"<i>NOT: Sadece son 1-2 mumda oluşmuş bullish kesişimler.</i>"
    )
    send_telegram_message(header)

    if bist_text:
        send_telegram_message(bist_text)
    
    if nasdaq_text:
        send_telegram_message(nasdaq_text)
    
    send_telegram_message(crypto_text)

    # TCE skorları en son ayrı mesaj olarak (plain text)
    if tce_text:
        send_telegram_message(tce_text, parse_mode=None)


if __name__ == "__main__":
    main()
