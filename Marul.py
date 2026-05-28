"""
Marul Buyume Takibi — Interaktif Kontur Duzenleyici
====================================================
Cikti: marul_buyume.csv  (zaman, alan_cm2)
       marul_buyume.png  (buyume grafigi)

Calisma modu:
  1. Otomatik:  Yesil maske ile marul konturu tespit edilir.
  2. Interaktif: Kontur anchor noktalari ile gosterilir,
     kullanici noktalardan tutup surekleyerek siniri duzeltir.
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
import json

# ─────────────────────────────────────────────
# 0. AYARLAR
# ─────────────────────────────────────────────

RESIM_KLASORU = r"D:\Marul\2nd Experiment\Images\Daily_Images\cva\raspberry"
CIKTI_CSV     = "marul_buyume.csv"
CIKTI_GRAFIK  = "marul_buyume.png"

HSV_YESIL_ALT = np.array([25,  40,  40])
HSV_YESIL_UST = np.array([95, 255, 255])

MIN_KONTUR_ALANI = 500
KARANLIK_ESIGI   = 85
MIN_YESIL_ORAN   = 0.01
DEPTH_TOLERANS_MM = 80
SAKSI_ARALIK_MM   = 200

# Interaktif mod: Bu tarihten sonra kontur duzenleme penceresi acilir.
# None ise her zaman otomatik mod (duzenleme penceresi acilmaz).
INTERAKTIF_BASLANGIC = "2022-05-3"  # Ornek: "2022-05-12"

# Anchor noktasi ayarlari
ANCHOR_SAYISI    = 30    # Kontur uzerindeki kontrol noktasi sayisi
ANCHOR_YARICAP   = 7     # Anchor cizim yaricapi (piksel, gosterim olceginde)
YAKALAMA_MESAFE  = 15    # Fareyle bu mesafe icindeki anchor secilir

# ─────────────────────────────────────────────
# 1. YARDIMCI FONKSIYONLAR
# ─────────────────────────────────────────────

def dosya_adini_parse_et(dosya_adi: str):
    eslesme = re.search(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})', dosya_adi)
    if not eslesme:
        return None
    yil, ay, gun, saat, dakika, _ = map(int, eslesme.groups())
    try:
        return datetime(yil, ay, gun, saat, dakika, 0)
    except ValueError:
        return None


def depth_ve_intrinsik_oku(rgb_yol: Path):
    """RGB yanindaki _depth.png ve intr_extr.json dosyalarini okur."""
    stem = rgb_yol.stem.replace("_rgb", "")
    klasor = rgb_yol.parent

    depth_yol = klasor / f"{stem}_depth.png"
    if not depth_yol.exists():
        return None, None, None

    fx, fy, depth_scale = None, None, 0.001
    for suffix in ["intr_extr.json", "_intr_extr.json"]:
        aday = klasor / f"{stem}{suffix}"
        if aday.exists():
            try:
                with open(aday, "r") as f:
                    intr = json.load(f)
                depth_scale = intr.get("DepthScale", 0.001)
                ci = intr.get("color_int", intr.get("depth_int", {}))
                fx = ci.get("fx")
                fy = ci.get("fy")
            except Exception:
                pass
            break

    if fx is None or fy is None:
        for json_dosya in klasor.glob("*intr_extr.json"):
            try:
                with open(json_dosya, "r") as f:
                    intr = json.load(f)
                depth_scale = intr.get("DepthScale", 0.001)
                ci = intr.get("color_int", intr.get("depth_int", {}))
                fx = ci.get("fx", 1370.0)
                fy = ci.get("fy", 1368.0)
                break
            except Exception:
                continue
        if fx is None:
            fx, fy = 1370.0, 1368.0

    depth_raw = cv2.imread(str(depth_yol), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None, None, None

    depth_mm = depth_raw.astype(np.float32) * depth_scale * 1000.0
    return depth_mm, fx, fy


def kontur_bul(img):
    """Goruntuden en uygun marul konturunu bul."""
    h, w = img.shape[:2]
    mx, my = w // 2, h // 2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maske = cv2.inRange(hsv, HSV_YESIL_ALT, HSV_YESIL_UST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN,  kernel, iterations=2)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, kernel, iterations=2)

    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konturlar:
        return None, maske

    def skor(k):
        alan = cv2.contourArea(k)
        if alan < MIN_KONTUR_ALANI:
            return -1
        x, y, bw, bh = cv2.boundingRect(k)
        kx, ky = x + bw / 2, y + bh / 2
        d = ((kx - mx)**2 + (ky - my)**2) ** 0.5
        return alan / (1 + d * 0.1)

    en_iyi = max(konturlar, key=skor)
    if cv2.contourArea(en_iyi) < MIN_KONTUR_ALANI:
        return None, maske

    return en_iyi, maske


def kontur_to_anchors(kontur, n=ANCHOR_SAYISI):
    """Konturu n adet esit aralikli anchor noktasina donustur."""
    kontur = kontur.squeeze()
    if kontur.ndim != 2:
        return kontur.reshape(-1, 2)

    # Kontur uzunlugu boyunca esit aralikli noktalar sec
    toplam = len(kontur)
    if toplam <= n:
        return kontur.copy()

    indices = np.linspace(0, toplam - 1, n, dtype=int)
    return kontur[indices].copy()


def anchors_to_kontur(anchors, puan_sayisi=200):
    """Anchor noktalarindan duz kontur olustur (ara noktalari interpolasyon ile doldur)."""
    n = len(anchors)
    if n < 3:
        return anchors.reshape(-1, 1, 2).astype(np.int32)

    # Kapalı döngü için ilk noktayı sona ekle
    kapali = np.vstack([anchors, anchors[0:1]])

    # Her segment arasini interpolasyonla doldur
    noktalar = []
    segment_puan = max(puan_sayisi // n, 2)
    for i in range(n):
        p1 = kapali[i].astype(float)
        p2 = kapali[i + 1].astype(float)
        for t in np.linspace(0, 1, segment_puan, endpoint=False):
            noktalar.append(p1 + t * (p2 - p1))

    kontur = np.array(noktalar, dtype=np.int32).reshape(-1, 1, 2)
    return kontur


# ─────────────────────────────────────────────
# 2. INTERAKTIF KONTUR DUZENLEYICI
# ─────────────────────────────────────────────

class KonturDuzenleyici:
    """
    OpenCV penceresi ile kontur anchor noktalarini surekleyerek
    marul sinirini duzeltmeye izin veren interaktif arac (zoom + nokta duzenleme).

    Kontroller:
      Sol tik + surekle   :  Anchor noktasini tasi
      Sol CIFT tik        :  En yakin kenara yeni anchor EKLE
      Sag CIFT tik        :  Imlece en yakin anchor'i SIL
      Fare tekerlegi      :  Imlec etrafinda yakinlas / uzaklas
      Sag tik + surekle   :  Goruntuyu kaydir (pan)
      z / x               :  Klavyeyle yakinlas / uzaklas
      W A S D             :  Goruntuyu kaydir
      0                   :  Zoom'u sifirla (tam goruntu)
      ENTER               :  Onayla
      ESC                 :  Atla (NaN)
      R                   :  Otomatik konturu sifirla
      +/-                 :  Anchor sayisini topluca artir/azalt
    """

    PANEL_H   = 100    # Ust bilgi paneli yuksekligi
    KENAR     = 40     # Resmin etrafindaki bosluk
    HEDEF_W   = 1280   # Pencere hedef genisligi
    HEDEF_H   = 760    # Pencere hedef yuksekligi
    MAX_ZOOM  = 8.0    # En fazla yakinlasma
    ZOOM_ADIM = 1.2    # Her adimdaki zoom carpani
    MIN_ANCHOR = 3     # Silme sonrasi en az kalmasi gereken nokta sayisi

    def __init__(self, img_orijinal, kontur, dosya_adi=""):
        self.img_orijinal = img_orijinal
        self.dosya_adi = dosya_adi
        self.orijinal_kontur = kontur.copy()

        h, w = img_orijinal.shape[:2]
        kul_w = self.HEDEF_W - 2 * self.KENAR
        kul_h = self.HEDEF_H - self.PANEL_H - 2 * self.KENAR
        self.olcek = min(kul_w / w, kul_h / h, 1.0)
        self.g_w = int(w * self.olcek)
        self.g_h = int(h * self.olcek)
        self.img_gosterim = cv2.resize(img_orijinal, (self.g_w, self.g_h))

        self.anchors = (kontur_to_anchors(kontur) * self.olcek).astype(int)
        self.secili_idx = -1

        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self.pan_aktif = False
        self.pan_son = (0, 0)

    # ---- Koordinat donusumleri ----
    def _img_to_vp(self, ix, iy):
        return (ix * self.zoom - self.view_x, iy * self.zoom - self.view_y)

    def _vp_to_img(self, vx, vy):
        return ((vx + self.view_x) / self.zoom, (vy + self.view_y) / self.zoom)

    def _clamp_view(self):
        bw = self.g_w * self.zoom
        bh = self.g_h * self.zoom
        self.view_x = max(0.0, min(self.view_x, bw - self.g_w))
        self.view_y = max(0.0, min(self.view_y, bh - self.g_h))

    def _zoom_merkez(self, carpan, vx, vy):
        ix, iy = self._vp_to_img(vx, vy)
        self.zoom = max(1.0, min(self.zoom * carpan, self.MAX_ZOOM))
        self.view_x = ix * self.zoom - vx
        self.view_y = iy * self.zoom - vy
        self._clamp_view()

    # ---- Nokta ekleme / silme ----
    @staticmethod
    def _nokta_segment_mesafe(p, a, b):
        """p noktasinin [a, b] segmentine olan en kisa mesafesi (viewport px)."""
        p, a, b = np.asarray(p, float), np.asarray(a, float), np.asarray(b, float)
        ab = b - a
        uzun2 = float(ab @ ab)
        if uzun2 < 1e-9:
            return float(np.linalg.norm(p - a))
        t = max(0.0, min(1.0, float((p - a) @ ab) / uzun2))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _ara_nokta_ekle(self, vx, vy):
        """Tiklanan konuma en yakin kenar (segment) uzerine yeni anchor ekle."""
        n = len(self.anchors)
        if n < 2:
            return
        anc_vp = [self._img_to_vp(ax, ay) for ax, ay in self.anchors]
        p = (vx, vy)
        en_iyi_i, en_iyi_d = 0, float("inf")
        for i in range(n):
            a = anc_vp[i]
            b = anc_vp[(i + 1) % n]
            d = self._nokta_segment_mesafe(p, a, b)
            if d < en_iyi_d:
                en_iyi_d, en_iyi_i = d, i

        ix, iy = self._vp_to_img(vx, vy)
        yeni = [int(max(0, min(self.g_w - 1, ix))),
                int(max(0, min(self.g_h - 1, iy)))]
        self.anchors = np.insert(self.anchors, en_iyi_i + 1, yeni, axis=0)
        self.secili_idx = -1

    def _nokta_sil(self, vx, vy):
        """Imlece en yakin anchor'i sil (en az MIN_ANCHOR nokta kalir)."""
        if len(self.anchors) <= self.MIN_ANCHOR:
            return
        en_yakin, en_d = -1, float("inf")
        for i, (ax, ay) in enumerate(self.anchors):
            avx, avy = self._img_to_vp(ax, ay)
            d = ((vx - avx) ** 2 + (vy - avy) ** 2) ** 0.5
            if d < en_d:
                en_d, en_yakin = d, i
        if en_yakin >= 0 and en_d < YAKALAMA_MESAFE * 1.6:
            self.anchors = np.delete(self.anchors, en_yakin, axis=0)
            self.secili_idx = -1

    # ---- Cizim ----
    def _ciz(self):
        M, P = self.KENAR, self.PANEL_H

        bw = int(round(self.g_w * self.zoom))
        bh = int(round(self.g_h * self.zoom))
        buyuk = cv2.resize(self.img_gosterim, (bw, bh), interpolation=cv2.INTER_LINEAR)

        self._clamp_view()
        vx0 = int(round(self.view_x))
        vy0 = int(round(self.view_y))
        vp = buyuk[vy0:vy0 + self.g_h, vx0:vx0 + self.g_w].copy()
        if vp.shape[0] != self.g_h or vp.shape[1] != self.g_w:
            vp = cv2.copyMakeBorder(
                vp, 0, max(0, self.g_h - vp.shape[0]),
                0, max(0, self.g_w - vp.shape[1]),
                cv2.BORDER_CONSTANT, value=(30, 30, 30)
            )[:self.g_h, :self.g_w]

        anc_vp = np.array([self._img_to_vp(ax, ay) for ax, ay in self.anchors],
                          dtype=np.int32)
        kontur_vp = anchors_to_kontur(anc_vp)

        overlay = vp.copy()
        cv2.drawContours(overlay, [kontur_vp], -1, (0, 220, 0), -1)
        cv2.addWeighted(overlay, 0.25, vp, 0.75, 0, vp)
        cv2.drawContours(vp, [kontur_vp], -1, (0, 255, 0), 2)

        for i, (ax, ay) in enumerate(anc_vp):
            if -20 <= ax <= self.g_w + 20 and -20 <= ay <= self.g_h + 20:
                if i == self.secili_idx:
                    renk = (0, 0, 255); r = ANCHOR_YARICAP + 3
                else:
                    renk = (255, 255, 0); r = ANCHOR_YARICAP
                cv2.circle(vp, (int(ax), int(ay)), r, renk, -1)
                cv2.circle(vp, (int(ax), int(ay)), r, (255, 255, 255), 1)

        tuval = cv2.copyMakeBorder(vp, M, M, M, M,
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30))

        canvas_w = tuval.shape[1]
        panel = np.zeros((P, canvas_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

        kontur_orijinal = (self.anchors / self.olcek).astype(np.int32)
        alan_px = cv2.contourArea(anchors_to_kontur(kontur_orijinal))

        cv2.putText(panel, self.dosya_adi, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(panel,
                    f"Alan: {alan_px:.0f} px | Anchor: {len(self.anchors)} | "
                    f"Zoom: {self.zoom:.1f}x",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(panel,
                    "SolCiftTik=nokta EKLE | SagCiftTik=nokta SIL | "
                    "SolTik+surekle=tasi",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
        cv2.putText(panel,
                    "Tekerlek/z-x=zoom | SagTik veya WASD=kaydir | 0=sifirla | "
                    "ENTER ESC R +/-",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        return np.vstack([panel, tuval])

    # ---- Fare ----
    def _fare_callback(self, event, x, y, flags, param):
        M, P = self.KENAR, self.PANEL_H
        vx = x - M
        vy = y - (P + M)

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            carpan = self.ZOOM_ADIM if delta > 0 else 1.0 / self.ZOOM_ADIM
            self._zoom_merkez(carpan, vx, vy)
            cv2.imshow("Kontur Duzenle", self._ciz())
            return

        if event == cv2.EVENT_LBUTTONDBLCLK:   # nokta ekle
            if y < P:
                return
            self._ara_nokta_ekle(vx, vy)
            cv2.imshow("Kontur Duzenle", self._ciz())
            return

        if event == cv2.EVENT_RBUTTONDBLCLK:   # nokta sil
            if y < P:
                return
            self.pan_aktif = False
            self._nokta_sil(vx, vy)
            cv2.imshow("Kontur Duzenle", self._ciz())
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            self.pan_aktif = True
            self.pan_son = (x, y)
            return

        if event == cv2.EVENT_RBUTTONUP:
            self.pan_aktif = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if y < P:
                return
            for i, (ax, ay) in enumerate(self.anchors):
                avx, avy = self._img_to_vp(ax, ay)
                if ((vx - avx) ** 2 + (vy - avy) ** 2) ** 0.5 < YAKALAMA_MESAFE:
                    self.secili_idx = i
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.secili_idx >= 0:
                ix, iy = self._vp_to_img(vx, vy)
                self.anchors[self.secili_idx] = [
                    int(max(0, min(self.g_w - 1, ix))),
                    int(max(0, min(self.g_h - 1, iy))),
                ]
                cv2.imshow("Kontur Duzenle", self._ciz())
            elif self.pan_aktif:
                lx, ly = self.pan_son
                self.view_x -= (x - lx)
                self.view_y -= (y - ly)
                self.pan_son = (x, y)
                self._clamp_view()
                cv2.imshow("Kontur Duzenle", self._ciz())

        elif event == cv2.EVENT_LBUTTONUP:
            self.secili_idx = -1

    # ---- Calistir ----
    def calistir(self):
        cv2.namedWindow("Kontur Duzenle", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Kontur Duzenle", self._fare_callback)
        cv2.imshow("Kontur Duzenle", self._ciz())

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == 13:  # ENTER -> onayla
                cv2.destroyWindow("Kontur Duzenle")
                orijinal_anchors = (self.anchors / self.olcek).astype(np.int32)
                return anchors_to_kontur(orijinal_anchors)

            elif key == 27:  # ESC -> atla
                cv2.destroyWindow("Kontur Duzenle")
                return None

            elif key == ord('r'):  # konturu sifirla
                self.anchors = (kontur_to_anchors(self.orijinal_kontur) * self.olcek).astype(int)
                cv2.imshow("Kontur Duzenle", self._ciz())

            elif key == ord('0'):  # zoom sifirla
                self.zoom = 1.0
                self.view_x = self.view_y = 0.0
                cv2.imshow("Kontur Duzenle", self._ciz())

            elif key == ord('z'):
                self._zoom_merkez(self.ZOOM_ADIM, self.g_w / 2, self.g_h / 2)
                cv2.imshow("Kontur Duzenle", self._ciz())

            elif key == ord('x'):
                self._zoom_merkez(1.0 / self.ZOOM_ADIM, self.g_w / 2, self.g_h / 2)
                cv2.imshow("Kontur Duzenle", self._ciz())

            elif key in (ord('w'), ord('a'), ord('s'), ord('d')):
                adim = 60
                if key == ord('w'): self.view_y -= adim
                elif key == ord('s'): self.view_y += adim
                elif key == ord('a'): self.view_x -= adim
                elif key == ord('d'): self.view_x += adim
                self._clamp_view()
                cv2.imshow("Kontur Duzenle", self._ciz())

            elif key == ord('+') or key == ord('='):
                n = min(len(self.anchors) + 5, 80)
                kontur_orig = (self.anchors / self.olcek).astype(np.int32)
                tam_kontur = anchors_to_kontur(kontur_orig)
                self.anchors = (kontur_to_anchors(tam_kontur.squeeze(), n) * self.olcek).astype(int)
                cv2.imshow("Kontur Duzenle", self._ciz())

            elif key == ord('-') or key == ord('_'):
                n = max(len(self.anchors) - 5, 8)
                kontur_orig = (self.anchors / self.olcek).astype(np.int32)
                tam_kontur = anchors_to_kontur(kontur_orig)
                self.anchors = (kontur_to_anchors(tam_kontur.squeeze(), n) * self.olcek).astype(int)
                cv2.imshow("Kontur Duzenle", self._ciz())

        cv2.destroyWindow("Kontur Duzenle")
        return None
# ─────────────────────────────────────────────
# 3. ALAN HESAPLAMA
# ─────────────────────────────────────────────

def alan_cm2_hesapla(yol: Path, interaktif=False):
    """
    Bir resimden merkez marulun alanini cm2 cinsinden hesaplar.
    interaktif=True ise kontur duzenleme penceresi acilir.
    """
    img = cv2.imread(str(yol))
    if img is None:
        return None

    h, w = img.shape[:2]
    mx, my = w // 2, h // 2

    # Karanlik filtresi
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 2].mean() < KARANLIK_ESIGI:
        return None

    # Kontur bul
    kontur, maske = kontur_bul(img)

    yesil_oran = cv2.countNonZero(maske) / (w * h)
    if yesil_oran < MIN_YESIL_ORAN:
        return None

    if kontur is None:
        return None

    # ── Interaktif mod ──────────────────────
    if interaktif:
        duzenleyici = KonturDuzenleyici(img, kontur, yol.name)
        sonuc_kontur = duzenleyici.calistir()
        if sonuc_kontur is None:
            return None  # Kullanici atladi

        # Duzenlenmis konturdan maske olustur
        maske = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(maske, [sonuc_kontur], -1, 255, -1)
    else:
        # Otomatik mod: dairesel ROI ile sinirla
        pass

    # ── Depth tabanli gercek alan hesapla ──
    depth_mm, fx, fy = depth_ve_intrinsik_oku(yol)

    if depth_mm is None:
        return None

    if depth_mm.shape[:2] != (h, w):
        depth_mm = cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_NEAREST)

    # Merkez bolgeden referans depth
    r = 25
    y1, y2 = max(0, my - r), min(h, my + r)
    x1, x2 = max(0, mx - r), min(w, mx + r)
    bolge_d = depth_mm[y1:y2, x1:x2]
    bolge_m = maske[y1:y2, x1:x2]

    gecerli = (bolge_m > 0) & (bolge_d > 0)
    if gecerli.sum() < 5:
        tum = (maske > 0) & (depth_mm > 0)
        if tum.sum() < 10:
            return None
        ref_depth = np.median(depth_mm[tum])
    else:
        ref_depth = np.median(bolge_d[gecerli])

    if ref_depth <= 0:
        return None

    if not interaktif:
        # Otomatik modda dairesel ROI
        mm_per_px = ref_depth / fx
        roi_r = int((SAKSI_ARALIK_MM / 2) / mm_per_px)
        yc, xc = np.ogrid[:h, :w]
        roi_circle = (xc - mx)**2 + (yc - my)**2 <= roi_r**2
    else:
        # Interaktif modda kontur maskesi zaten uygulanmis
        roi_circle = np.ones((h, w), dtype=bool)

    # Depth filtresi
    df = (depth_mm >= ref_depth - DEPTH_TOLERANS_MM) & \
         (depth_mm <= ref_depth + DEPTH_TOLERANS_MM) & \
         (depth_mm > 0)

    son = (maske > 0) & roi_circle & df
    piksel_depth = depth_mm[son]
    if len(piksel_depth) == 0:
        return None

    alan_mm2 = ((piksel_depth / fx) * (piksel_depth / fy)).sum()
    alan_cm2 = alan_mm2 / 100.0

    return round(float(alan_cm2), 2)


# ─────────────────────────────────────────────
# 4. GRAFIK
# ─────────────────────────────────────────────

def grafik_olustur(df, cikti):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Marul Buyume Takibi", fontsize=16, fontweight="bold")

    x = df["zaman"]
    y = df["alan_cm2"]

    gecerli = y.notna()
    ax.scatter(x[gecerli], y[gecerli], color="#81C784", s=8, alpha=0.4,
               label="Saatlik olcum", zorder=2)

    df_tmp = df[["zaman", "alan_cm2"]].copy()
    df_tmp["gun"] = df_tmp["zaman"].dt.date
    gunluk = df_tmp.groupby("gun")["alan_cm2"].median().reset_index()
    gunluk["gun"] = pd.to_datetime(gunluk["gun"])
    ax.plot(gunluk["gun"], gunluk["alan_cm2"], color="#2E7D32", linewidth=2.5,
            marker="o", markersize=5, label="Gunluk medyan (cm2)", zorder=3)
    ax.fill_between(gunluk["gun"], gunluk["alan_cm2"], alpha=0.15, color="#4CAF50")

    ax.set_ylabel("Alan (cm2)", fontsize=12)
    ax.set_xlabel("Zaman", fontsize=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(cikti, dpi=150, bbox_inches="tight")
    print(f"\n[OK] Grafik kaydedildi -> {cikti}")
    plt.close()


# ─────────────────────────────────────────────
# 5. ANA PROGRAM
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MARUL BUYUME TAKIBI — Interaktif Kontur Duzenleyici")
    print("=" * 55)

    klasor = Path(RESIM_KLASORU)
    uzantilar = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    dosyalar = sorted([
        p for p in klasor.iterdir()
        if p.suffix.lower() in uzantilar and p.stem.endswith("_rgb")
    ])

    if not dosyalar:
        sys.exit(f"[HATA] '{RESIM_KLASORU}' klasorunde resim bulunamadi.")

    print(f"Toplam {len(dosyalar)} resim bulundu.\n")

    interaktif_tarih = None
    if INTERAKTIF_BASLANGIC:
        interaktif_tarih = datetime.strptime(INTERAKTIF_BASLANGIC, "%Y-%m-%d")
        print(f"[!] Interaktif mod: {INTERAKTIF_BASLANGIC} tarihinden sonra aktif")
        print("    Kontur uzerindeki noktalardan tutup surekleyin.")
        print("    ENTER=onayla | ESC=atla | R=sifirla | +/-=nokta sayisi\n")

    kayitlar = []

    for dosya in dosyalar:
        zaman = dosya_adini_parse_et(dosya.stem)
        if zaman is None:
            continue

        inter = False
        if interaktif_tarih and zaman >= interaktif_tarih:
            inter = True

        print(f"  {dosya.name}  [{zaman.strftime('%Y-%m-%d %H:%M')}]", end="")

        alan = alan_cm2_hesapla(dosya, interaktif=inter)

        if alan is not None:
            print(f"  -> {alan:.1f} cm2")
        else:
            print(f"  -> NaN")

        kayitlar.append({"zaman": zaman, "alan_cm2": alan})

    if not kayitlar:
        sys.exit("[HATA] Hicbir resim analiz edilemedi.")

    df = pd.DataFrame(kayitlar).sort_values("zaman").reset_index(drop=True)

    # Outlier filtresi
    pencere = 13
    g = df["alan_cm2"].dropna()
    if len(g) > pencere:
        g_med = g.rolling(window=pencere, center=True, min_periods=3).median()
        df.loc[g.index, "_med"] = g_med
        outlier = (
            df["alan_cm2"].notna() &
            df["_med"].notna() &
            (
                (df["alan_cm2"] < df["_med"] * 0.60) |
                (df["alan_cm2"] > df["_med"] * 1.50)
            )
        )
        if outlier.any():
            print(f"\n  [filtre] {outlier.sum()} outlier temizlendi")
            df.loc[outlier, "alan_cm2"] = np.nan
        df.drop(columns=["_med"], inplace=True, errors="ignore")

    # CSV kaydet
    df_csv = df.copy()
    df_csv["zaman"] = df_csv["zaman"].dt.strftime("%Y-%m-%d %H:%M")
    df_csv.to_csv(CIKTI_CSV, index=False, encoding="utf-8-sig")

    gecerli = df["alan_cm2"].notna().sum()
    print(f"\n[OK] CSV kaydedildi -> {CIKTI_CSV}")
    print(f"   Toplam: {len(df)} kayit ({gecerli} gecerli)")
    print(f"   Zaman : {df['zaman'].iloc[0]} -> {df['zaman'].iloc[-1]}")
    if gecerli > 0:
        print(f"   Min   : {df['alan_cm2'].min():.1f} cm2")
        print(f"   Max   : {df['alan_cm2'].max():.1f} cm2")

    grafik_olustur(df, CIKTI_GRAFIK)
    print("\nTamamlandi!")