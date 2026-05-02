"""
Marul Büyüme Takibi - Görüntü Analizi
======================================
Dosya adı formatı: d415_12_2022_05_03_19_00_02_rgb.jpg
Çıktı: marul_analiz.csv + marul_grafik.png
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import re
import sys

# ─────────────────────────────────────────────
# 0. AYARLAR — Sadece bu bölümü düzenle
# ─────────────────────────────────────────────

RESIM_KLASORU = r"D:\Marul\2nd Experiment\Images\Daily_Images\cva\raspberry"          # Resimlerinin bulunduğu klasör yolu
CIKTI_CSV     = "marul_analiz.csv"   # Çıktı CSV dosyası
CIKTI_GRAFIK  = "marul_grafik.png"   # Çıktı grafik dosyası

# HSV Yeşil renk aralığı (gerekirse ayarla)
# Soluk/sarımsı yeşil için alt sınırı düşür, koyu yeşil için üst sınırı artır
HSV_YESIL_ALT = np.array([25,  40,  40])   # (H, S, V) minimum
HSV_YESIL_UST = np.array([95, 255, 255])   # (H, S, V) maximum

# Gürültü temizleme (piksel) — küçük alanları eler
MIN_KONTUR_ALANI = 500

# Karanlık kare eşiği (0-255) — bu değerin altındaki ortalama parlaklık = gece/alacakaranlık
KARANLIK_ESIGI = 85

# Görüntü piksellerinin en az bu oranı kadar yeşil alan bulunamazsa NaN bas (0.0-1.0)
MIN_YESIL_ORAN = 0.01   # %1

# ─────────────────────────────────────────────
# 1. DOSYA ADI → ZAMAN PARSE
# ─────────────────────────────────────────────

def dosya_adini_parse_et(dosya_adi: str) -> datetime | None:
    """
    'd415_12_2022_05_03_19_00_02_rgb' formatından datetime üretir.
    Tanınamayan dosyaları None döndürerek atlar.
    """
    # Rakam gruplarını bul: yıl(4), ay(2), gün(2), saat(2), dk(2), sn(2)
    eslesme = re.search(
        r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})',
        dosya_adi
    )
    if not eslesme:
        return None
    yil, ay, gun, saat, dakika, _saniye = map(int, eslesme.groups())
    try:
        return datetime(yil, ay, gun, saat, dakika, 0)   # saniyeler atılır
    except ValueError:
        return None

def cihaz_id_parse_et(dosya_adi: str) -> str:
    """'d415_12_...' → 'd415_12' gibi cihaz önekini döndürür."""
    eslesme = re.match(r'^([a-zA-Z0-9]+_\d+)', dosya_adi)
    return eslesme.group(1) if eslesme else "bilinmiyor"

# ─────────────────────────────────────────────
# 2. TEK RESİM ANALİZİ
# ─────────────────────────────────────────────

def resim_analiz_et(yol: Path) -> dict | None:
    """
    Bir resmi okur; marulu tespit eder; boyut ve renk bilgilerini döndürür.
    """
    img = cv2.imread(str(yol))
    if img is None:
        print(f"  [!] Okunamadı: {yol.name}")
        return None

    goruntu_y, goruntu_x = img.shape[:2]
    merkez_x, merkez_y = goruntu_x // 2, goruntu_y // 2

    # ── Karanlık Kare Filtresi ────────────────
    # V kanalı (parlaklık) ortalaması düşükse gece karesi → NaN
    hsv_on = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ort_parlaklik = hsv_on[:, :, 2].mean()
    if ort_parlaklik < KARANLIK_ESIGI:
        print(f"  [gece]  Parlaklık={ort_parlaklik:.1f} → NaN")
        return None

    # ── Yeşil Maske ──────────────────────────
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, HSV_YESIL_ALT, HSV_YESIL_UST)

    # Morfolojik temizlik (gürültü ve delik kapatma)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN,  kernel, iterations=2)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ── Kontur Bulma ─────────────────────────
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Toplam yeşil piksel oranı — çok düşükse kare güvenilir değil
    yesil_piksel_orani = cv2.countNonZero(maske) / (goruntu_x * goruntu_y)
    if yesil_piksel_orani < MIN_YESIL_ORAN:
        print(f"  [az-yeşil] Yeşil oran={yesil_piksel_orani:.4f} → NaN")
        return None

    if not konturlar:
        print(f"  [!] Kontur bulunamadı: {yol.name}")
        return None

    # Merkeze en yakın + en büyük konturu seç
    def skor(k):
        alan = cv2.contourArea(k)
        if alan < MIN_KONTUR_ALANI:
            return -1
        x, y, w, h = cv2.boundingRect(k)
        kx, ky = x + w / 2, y + h / 2
        merkez_uzakligi = ((kx - merkez_x)**2 + (ky - merkez_y)**2) ** 0.5
        # Büyük alan + merkeze yakın → yüksek skor
        return alan / (1 + merkez_uzakligi * 0.1)

    marul_konturu = max(konturlar, key=skor)
    if cv2.contourArea(marul_konturu) < MIN_KONTUR_ALANI:
        print(f"  [!] Yeterli büyüklükte kontur yok: {yol.name}")
        return None

    # ── Boyut Hesaplama ───────────────────────
    x, y, w, h = cv2.boundingRect(marul_konturu)
    piksel_alani = cv2.contourArea(marul_konturu)  # gerçek kontur alanı (daha doğru)

    # ── Renk Yoğunluğu ───────────────────────
    marul_maskesi = np.zeros(maske.shape, dtype=np.uint8)
    cv2.drawContours(marul_maskesi, [marul_konturu], -1, 255, thickness=cv2.FILLED)

    ortalama = cv2.mean(img, mask=marul_maskesi)  # BGR sırası
    b_ort, g_ort, r_ort = ortalama[0], ortalama[1], ortalama[2]

    # Greenness Index: yeşil kanalın kırmızı ve mavi ortalamasına oranı
    gb_ort = (r_ort + b_ort) / 2 if (r_ort + b_ort) > 0 else 1
    yesil_indeksi = g_ort / gb_ort  # >1 = canlı yeşil, <1 = solar/sarı

    return {
        "en_piksel":      w,
        "boy_piksel":     h,
        "alan_piksel":    round(piksel_alani, 1),
        "r_kanal":        round(r_ort, 1),
        "g_kanal":        round(g_ort, 1),
        "b_kanal":        round(b_ort, 1),
        "yesil_indeksi":  round(yesil_indeksi, 4),
    }

# ─────────────────────────────────────────────
# 3. TÜM KLASÖRÜ İŞLE
# ─────────────────────────────────────────────

def klasoru_isle(klasor: str) -> pd.DataFrame:
    uzantilar = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    dosyalar  = sorted([
        p for p in Path(klasor).iterdir()
        if p.suffix.lower() in uzantilar and p.stem.endswith("_rgb")
    ])
    if not dosyalar:
        sys.exit(f"[HATA] '{klasor}' klasöründe resim bulunamadı.")

    print(f"Toplam {len(dosyalar)} resim bulundu. İşleniyor...\n")
    kayitlar = []

    for dosya in dosyalar:
        zaman = dosya_adini_parse_et(dosya.stem)
        if zaman is None:
            print(f"  [~] Atlandı (tarih parse edilemedi): {dosya.name}")
            continue

        print(f"  Analiz ediliyor → {dosya.name}  [{zaman}]")
        sonuc = resim_analiz_et(dosya)
        cihaz = cihaz_id_parse_et(dosya.stem)
        if sonuc:
            kayitlar.append({"zaman": zaman, "cihaz": cihaz, "dosya": dosya.name, **sonuc})
        else:
            kayitlar.append({
                "zaman": zaman, "cihaz": cihaz, "dosya": dosya.name,
                "en_piksel": np.nan, "boy_piksel": np.nan,
                "alan_piksel": np.nan,
                "r_kanal": np.nan, "g_kanal": np.nan, "b_kanal": np.nan,
                "yesil_indeksi": np.nan,
            })

    if not kayitlar:
        sys.exit("[HATA] Hiçbir resim analiz edilemedi.")

    df = pd.DataFrame(kayitlar).sort_values("zaman").reset_index(drop=True)

    # ── Rolling Outlier Filtresi ─────────────
    # Sadece geçerli (NaN olmayan) değerler üzerinde rolling medyan hesapla.
    # Böylece gece boşlukları pencereyi seyreltmez; fide dönemi korunur çünkü
    # komşuları da küçük değerler içerir.
    pencere = 13   # saatlik rgb çekimde 13 nokta ≈ ±6 saat
    gecerli_idx  = df["alan_piksel"].dropna().index
    gecerli_vals = df.loc[gecerli_idx, "alan_piksel"]
    rol_med = gecerli_vals.rolling(window=pencere, center=True, min_periods=3).median()
    df.loc[gecerli_idx, "_rol_med"] = rol_med

    dip_mask = (
        df["alan_piksel"].notna() &
        df["_rol_med"].notna() &
        (df["alan_piksel"] < df["_rol_med"] * 0.85)
    )
    if dip_mask.any():
        print(f"  [rolling-dip] {dip_mask.sum()} ani dip → NaN")
        for col in ["en_piksel","boy_piksel","alan_piksel",
                    "r_kanal","g_kanal","b_kanal","yesil_indeksi"]:
            df.loc[dip_mask, col] = np.nan
    df.drop(columns=["_rol_med"], inplace=True, errors="ignore")

    # ── Göreceli Büyüme Oranı ───────────────
    # Baz: geçerli değerlerin medyanı (başlangıç gürültüsüne karşı dayanıklı)
    baz_alan = df["alan_piksel"].median()
    df["buyume_orani_%"] = ((df["alan_piksel"] / baz_alan) * 100).round(2)

    return df

# ─────────────────────────────────────────────
# 4. GRAFİK
# ─────────────────────────────────────────────

def grafik_olustur(df: pd.DataFrame, cikti: str):
    fig, eksenler = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Marul Büyüme Takibi", fontsize=16, fontweight="bold", y=0.98)
    renk_mavi   = "#2196F3"
    renk_yesil  = "#4CAF50"
    renk_turuncu = "#FF9800"

    x = df["zaman"]

    def nan_fill(ax, x, y, label=None, **kwargs):
        """NaN içeren seride fill_between'in 0'a çekilmesini önler.
        Legend yalnızca ilk geçerli bloğa eklenir."""
        y = pd.Series(y.values, index=range(len(y)))
        bloklar = (~y.isna()).astype(int).diff().fillna(1)
        baslangiclar = bloklar[bloklar == 1].index.tolist()
        bitisler     = bloklar[bloklar == -1].index.tolist()
        if len(y) > 0 and not (y.iloc[-1] != y.iloc[-1]):
            bitisler.append(len(y))
        for i, (bas, bit) in enumerate(zip(baslangiclar, bitisler)):
            ax.fill_between(
                x.iloc[bas:bit], y.iloc[bas:bit],
                label=label if i == 0 else "_nolegend_",
                **kwargs
            )

    # — Alt grafik 1: Alan (büyüme) ——————————
    eksenler[0].plot(x, df["alan_piksel"], color=renk_mavi, linewidth=2, marker="o", markersize=4, label="Alan (piksel²)")
    nan_fill(eksenler[0], x, df["alan_piksel"], alpha=0.15, color=renk_mavi)
    eksenler[0].set_ylabel("Alan (piksel²)", fontsize=11)
    eksenler[0].set_title("Bitki Alanı (Ham Piksel)", fontsize=12)
    eksenler[0].legend(loc="upper left")
    eksenler[0].grid(True, linestyle="--", alpha=0.5)

    # — Alt grafik 2: Göreceli büyüme (%) ————
    eksenler[1].plot(x, df["buyume_orani_%"], color=renk_turuncu, linewidth=2, marker="s", markersize=4, label="Büyüme Oranı (%)")
    eksenler[1].axhline(100, color="gray", linestyle=":", linewidth=1, label="Medyan (100%)")
    buyume = df["buyume_orani_%"].copy()
    buyume_pozitif = buyume.where(buyume >= 100)
    nan_fill(eksenler[1], x, buyume_pozitif, alpha=0.15, color=renk_turuncu, label="Pozitif büyüme")
    eksenler[1].set_ylabel("Büyüme Oranı (%)", fontsize=11)
    eksenler[1].set_title("Göreceli Büyüme Oranı (Medyan = %100)", fontsize=12)
    eksenler[1].legend(loc="upper left")
    eksenler[1].grid(True, linestyle="--", alpha=0.5)

    # — Alt grafik 3: Yeşil İndeksi ————————
    eksenler[2].plot(x, df["yesil_indeksi"], color=renk_yesil, linewidth=2, marker="^", markersize=4, label="Yeşil İndeksi (G/((R+B)/2))")
    eksenler[2].axhline(1.0, color="gray", linestyle=":", linewidth=1, label="Nötr (1.0)")
    yesil_pozitif = df["yesil_indeksi"].where(df["yesil_indeksi"] >= 1.0)
    nan_fill(eksenler[2], x, yesil_pozitif, alpha=0.15, color=renk_yesil, label="Canlı yeşil bölge")
    eksenler[2].set_ylabel("Yeşil İndeksi", fontsize=11)
    eksenler[2].set_title("Renk Sağlık İndeksi (>1.0 = Canlı Yeşil)", fontsize=12)
    eksenler[2].set_xlabel("Zaman", fontsize=11)
    eksenler[2].legend(loc="upper left")
    eksenler[2].grid(True, linestyle="--", alpha=0.5)

    # X ekseni: tarih + saat formatı
    for ax in eksenler:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))

    plt.tight_layout()
    plt.savefig(cikti, dpi=150, bbox_inches="tight")
    print(f"\n✅ Grafik kaydedildi → {cikti}")
    plt.close()

# ─────────────────────────────────────────────
# 5. ANA PROGRAM
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MARUL BÜYÜME TAKİBİ — Görüntü Analizi")
    print("=" * 55)

    df = klasoru_isle(RESIM_KLASORU)

    # CSV için zaman sütununu saniyesiz string'e çevir (tablolarla join uyumu)
    df_csv = df.copy()
    df_csv["zaman"] = df_csv["zaman"].dt.strftime("%Y-%m-%d %H:%M")
    df_csv.to_csv(CIKTI_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ CSV kaydedildi   → {CIKTI_CSV}")
    gecerli = df["alan_piksel"].notna()
    print(f"   Toplam kayıt    : {len(df)}  ({gecerli.sum()} geçerli, {(~gecerli).sum()} NaN)")
    if "cihaz" in df.columns:
        print(f"   Cihaz dağılımı  : {df['cihaz'].value_counts().to_dict()}")
    print(f"   Zaman aralığı   : {df['zaman'].iloc[0]}  →  {df['zaman'].iloc[-1]}")
    son_gecerli = df.loc[gecerli, "buyume_orani_%"].iloc[-1]
    print(f"   Toplam büyüme   : %{son_gecerli:.1f}")
    print(f"   Ort. Yeşil İnd. : {df['yesil_indeksi'].mean(skipna=True):.3f}")

    grafik_olustur(df, CIKTI_GRAFIK)

    print("\nTamamlandı! 🌱")