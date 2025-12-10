import re
import requests
import pandas as pd

# BIST Tüm (XUTUM) verileri - Oyak Yatırım sayfası
URL = "https://www.oyakyatirim.com.tr/piyasa-verileri/XUTUM"
OUTPUT_FILE = "bist_all.txt"


def fetch_all_bist_symbols_from_oyak():
    """
    Oyak Yatırım XUTUM sayfasından tüm BIST hisse kodlarını çeker.
    Örn: AEFES, AKBNK, THYAO ...
    Sonuçları .IS uzantılı hale getirir: AEFES.IS, AKBNK.IS ...
    """
    print(f"[*] Sayfa indiriliyor: {URL}")
    r = requests.get(URL, timeout=30)
    r.raise_for_status()
    html = r.text

    print("[*] Tablolar okunuyor (pandas.read_html)...")
    tables = pd.read_html(html)

    symbols = set()

    # Türkçe karakterleri de kapsayan basit bir pattern:
    # 3-6 büyük harf (BIST kodları genelde böyle)
    pattern = re.compile(r"^[A-ZÇĞİÖŞÜ]{3,6}$")

    for df in tables:
        # Sembol kolonu olan tabloyu bul
        for col in df.columns:
            col_name = str(col).strip().lower()
            if "sembol" in col_name or "symbol" in col_name:
                for v in df[col]:
                    s = str(v).strip()
                    if pattern.fullmatch(s):
                        symbols.add(s)

    symbols = sorted(symbols)
    print(f"[*] Toplam {len(symbols)} adet ham sembol bulundu.")

    yf_symbols = [s + ".IS" for s in symbols]
    return yf_symbols


def main():
    try:
        yf_symbols = fetch_all_bist_symbols_from_oyak()
    except Exception as e:
        print("Hata oluştu:", e)
        return

    if not yf_symbols:
        print("Hiç sembol bulunamadı, dosya yazılmadı.")
        return

    print(f"[*] {OUTPUT_FILE} dosyasına yazılıyor...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sym in yf_symbols:
            f.write(sym + "\n")

    print(f"[✓] İşlem bitti. {OUTPUT_FILE} oluşturuldu ({len(yf_symbols)} sembol).")


if __name__ == "__main__":
    main()
