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

TIMEFRAME_DAYS = "1d"  # Günlük mum (yfinance & Bybit 1D)

# ---- BIST evreni (havuz + likiditeye göre TOP N) ----
# İçine 500+ BIST hissesini yazacağımız havuz dosyası
BIST_ALL_FILE = os.getenv("BIST_ALL_FILE", "bist_all.txt")

# Havuzdan en likit kaç hisse taransın? (default: 150)
BIST_MAX_COUNT = int(os.getenv("BIST_MAX_COUNT", "150"))

# Mesajlarda gözükecek label
# Örn: "BIST Top 150 Likit"
BIST_LABEL = os.getenv("BIST_LABEL", f"BIST Top {BIST_MAX_COUNT} Likit")

# Kripto tarafı: Binance sembol listesi dosyası (BTCUSDT, ETHUSDT, ...)
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
        # Hata olursa fallback: tüm sembolleri aynen döndür
        return symbols

    multi = isinstance(data.columns, pd.MultiIndex)
    liquidity_list = []

    for sym in symbols:
        try:
            if multi:
                if sym not in data.columns.levels[0]:
                    # Bu sembol için veri yok
                    continue
                df_sym = data[sym].dropna()
            else:
                # Tek sembol durumu
                df_sym = data

            if df_sym.empty:
                continue

            # Gerekli kolonlar yoksa atla
            if "Close" not in df_sym.columns or "Volume" not in df_sym.columns:
                continue

            # Son 60 barı alsak yeterli
            df_recent = df_sym.tail(60)
            if len(df_recent) < min_days:
                # çok az veri, sağlıklı bir ortalama değil
                continue

            # Ortalama işlem değeri (TL): Close * Volume
            avg_value = (df_recent["Close"] * df_recent["Volume"]).mean()

            if pd.isna(avg_value) or avg_value <= 0:
                continue

            liquidity_list.append((sym, avg_value))

        except Exception as e:
            print(f"Likidite hesap hatası {sym}: {e}")
            continue

    if not liquidity_list:
        # Hiç veri alamadıysak fallback
        print(f"{universe_name} için likidite listesi boş, fallback ile tüm semboller kullanılacak.")
        return symbols

    # En yüksekten en düşüğe sırala
    liquidity_list.sort(key=lambda x: x[1], reverse=True)

    # İlk max_count kadarını al
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
    max_bars_ago: int = 1,   # en fazla kaç bar önce? 1 = son bar veya bir önceki bar
    max_days_ago: int = 2,   # en fazla kaç takvim günü önce?
    min_rel_gap: float = 0.0 # cross anında min fark (gap/price), 0 ise kontrol yok
) -> bool:
    """
    EMA fast & slow için bullish cross noktalarını bulur.

    Şartlar:
      - Cross, son bar veya ondan en fazla max_bars_ago bar önce olacak.
      - Cross'un tarihi bugünden en fazla max_days_ago gün önce olacak.
      - Eğer min_rel_gap > 0 ise: cross barında EMA_fast - EMA_slow,
        fiyata oranla en az min_rel_gap olmalı (çok ufak kesişimleri elemek için).
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

    # 1) Bar bazlı kontrol: son bar veya bir önceki bar içinde mi?
    if last_cross < last_idx - max_bars_ago:
        return False

    # 1.5) Gap kontrolü (isteğe bağlı)
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

    # 2) Tarih bazlı kontrol: cross barının tarihi bugünden max_days_ago günden eski olmasın
    idx = close.index
    if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
        try:
            last_cross_time = idx[last_cross]

            # PeriodIndex ise timestamp'e çevir
            if isinstance(last_cross_time, pd.Period):
                last_cross_time = last_cross_time.to_timestamp()

            # timezone'lu ise UTC'ye çevir, sonra naive yap
            if getattr(last_cross_time, "tzinfo", None) is not None:
                last_cross_time = last_cross_time.tz_convert("UTC").tz_localize(None)

            # Bugünün UTC tarihi (saat silinmiş)
            today_utc = pd.Timestamp.utcnow().normalize()
            cross_day = pd.Timestamp(last_cross_time).normalize()
            days_diff = (today_utc - cross_day).days

            if days_diff > max_days_ago:
                return False
        except Exception as e:
            # Tarih dönüşümünde hata olursa sadece bar filtresine göre karar verir
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


# =============== Kripto: Binance listesi, Bybit datası ===============

def map_binance_to_bybit_symbol(binance_symbol: str, markets: dict) -> str | None:
    """
    Binance sembolü (BTCUSDT, ETHUSDT, ARBUSDT ...) alır,
    Bybit'teki muhtemel market adlarına map etmeye çalışır.

    Öncelik:
      1) BTC/USDT:USDT (perpetual)
      2) BTC/USDT     (spot)
    """
    s = binance_symbol.strip().upper()
    if not s:
        return None

    if s.endswith("USDT"):
        base = s[:-4]
    else:
        base = s

    candidates = [
        f"{base}/USDT:USDT",  # USDT perpetual
        f"{base}/USDT",       # spot
    ]

    for c in candidates:
        if c in markets:
            return c

    return None


def scan_crypto_bybit_from_file(
    symbol_file: str = BINANCE_LIST_FILE,
    timeframe: str = "1d",
    limit: int = 400
):
    """
    Binance sembol listesi dosyasını (BTCUSDT, ETHUSDT, ...) okuyup,
    Bybit 1D mumları ile EMA13-34 / EMA34-89 bullish cross arar.
    """
    result = {
        "13_34_bull": [],
        "34_89_bull": [],
        "errors": [],
        "debug": ""
    }

    symbols = read_symbol_file(symbol_file)
    if not symbols:
        result["debug"] = f"{symbol_file} boş veya bulunamadı."
        return result

    try:
        exchange = ccxt.bybit({"enableRateLimit": True})
        exchange.load_markets()
        markets = exchange.markets
    except Exception as e:
        msg = f"Bybit borsası başlatılamadı: {e}"
        print(msg)
        result["errors"].append(msg)
        return result

    processed_count = 0

    for bin_sym in symbols:
        bin_sym_u = bin_sym.strip().upper()
        if not bin_sym_u:
            continue

        market_symbol = map_binance_to_bybit_symbol(bin_sym_u, markets)
        if market_symbol is None:
            print("Bybit market bulunamadı:", bin_sym_u)
            result["errors"].append(bin_sym_u)
            continue

        try:
            ohlcv = exchange.fetch_ohlcv(
                market_symbol,
                timeframe=timeframe,
                limit=limit,
            )
        except Exception as e:
            print("Bybit veri hatası:", bin_sym_u, market_symbol, "->", e)
            result["errors"].append(bin_sym_u)
            continue

        if not ohlcv or len(ohlcv) < 50:
            result["errors"].append(bin_sym_u)
            continue

        closes = pd.Series(
            [c[4] for c in ohlcv],
            index=pd.to_datetime([c[0] for c in ohlcv], unit="ms", utc=True),
        )

        try:
            if has_recent_bullish_cross(closes, 13, 34):
                result["13_34_bull"].append(bin_sym_u)

            if has_recent_bullish_cross(closes, 34, 89):
                result["34_89_bull"].append(bin_sym_u)

            processed_count += 1

        except Exception as e:
            print("Kripto hesap hatası:", bin_sym_u, "->", e)
            result["errors"].append(bin_sym_u)
            continue

    c13 = len(result["13_34_bull"])
    c34 = len(result["34_89_bull"])

    result["debug"] = (
        f"Kaynak: Bybit 1D. Binance listesinden {len(symbols)} sembol okundu, "
        f"geçerli veri: {processed_count}. "
        f"Sinyaller -> 13/34: {c13} adet, 34/89: {c34} adet."
    )

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
        f"Evren: {BIST_LABEL}, S&P 500, Seçili Kripto (Bybit)\n"
        f"NOT: Sadece son 1 mumda veya en fazla 2 mum önce oluşmuş bullish kesişimler listelenir."
    )
    send_telegram_message(header)

    # --- BIST (havuzdan likiditeye göre TOP N) --- #
    bist_all = read_symbol_file(BIST_ALL_FILE)

    if bist_all:
        # Havuzdan en likit BIST_MAX_COUNT hissenin seçilmesi
        bist_symbols = select_most_liquid_bist_symbols(
            bist_all,
            max_count=BIST_MAX_COUNT,
            universe_name="BIST Likit"
        )

        if bist_symbols:
            bist_res = scan_equity_universe(bist_symbols, "BIST Likit")
            # Gerçekte seçilen sayıyı label'a ve mesaja yansıtalım
            bist_label_full = f"{BIST_LABEL} ({len(bist_symbols)} hisse)"
            bist_text = format_result_block(f"🇹🇷 {bist_label_full}", bist_res)
            send_telegram_message(bist_text)
    else:
        print(f"{BIST_ALL_FILE} bulunamadı, BIST taraması yapılmayacak.")

    # --- S&P 500 (nasdaq100.txt dosyasından okunuyor) --- #
    sp500_symbols = read_symbol_file("nasdaq100.txt")
    if sp500_symbols:
        sp500_res = scan_equity_universe(sp500_symbols, "S&P 500")
        sp500_text = format_result_block("🇺🇸 S&P 500", sp500_res)
        send_telegram_message(sp500_text)

    # --- Kripto (Binance listesi, Bybit datası) --- #
    crypto_res = scan_crypto_bybit_from_file(symbol_file=BINANCE_LIST_FILE, timeframe="1d", limit=400)
    crypto_text = format_result_block("🪙 Kripto (Binance listesi, Bybit 1D)", crypto_res)
    send_telegram_message(crypto_text)

    dbg = crypto_res.get("debug")
    if dbg:
        send_telegram_message("🔍 " + dbg)


if __name__ == "__main__":
    main()
