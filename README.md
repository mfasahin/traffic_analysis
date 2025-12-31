# Traffic Analysis - Vehicle Counting and Density Analysis

Bu proje, video üzerinde araç sayımı ve yoğunluk analizi yapan bir Python uygulamasıdır. Clean Architecture prensiplerine uygun olarak geliştirilmiştir ve YOLOv8 ile OpenCV kullanmaktadır.

## Özellikler

- **🌐 Web Arayüzü** - Streamlit ile interaktif dashboard
- Araç tespiti (YOLOv8)
- Araç sayımı ve yoğunluk analizi
- ROI (Region of Interest) desteği - yol alanı tanımlama
- **Çoklu ROI desteği** - Birden fazla şerit analizi
- **Araç takibi (Vehicle Tracking)** - Aynı aracın birden fazla frame'de takibi
- **Hız hesaplama** - Piksel bazlı hız ölçümü ve km/h dönüşümü
- **Yön tespiti** - Araçların hareket yönü (yukarı/aşağı/sol/sağ)
- **Hız limiti ihlali tespiti** - Belirlenen hız limitini aşan araçların tespiti
- **Yön bazlı araç sayımı** - Her yöne giden araç sayısı
- **Araç boyutuna göre sınıflandırma** - Küçük/Orta/Büyük
- **Gece/Gündüz modu tespiti** - Otomatik tespit
- **Hava durumu analizi** - Yağmur, sis, kar, açık tespiti
- Detaylı istatistikler
- Video görselleştirme (hız, yön, tracking ID'leri)
- JSON formatında raporlama
- Clean Architecture mimarisi

## Kurulum

1. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

2. YOLOv8 modeli otomatik olarak indirilecektir (ilk çalıştırmada).

## 🌐 Web Arayüzü (ÖNERİLEN)

En kolay kullanım için web arayüzünü kullanın:

```bash
# Windows
run_web_app.bat

# Linux/Mac
chmod +x run_web_app.sh
./run_web_app.sh

# Veya direkt
streamlit run web_app.py
```

Web arayüzü açıldığında tarayıcınızda otomatik olarak açılacaktır (genellikle http://localhost:8501).

**Web Arayüzü Özellikleri:**
- ✅ Video yükleme ve önizleme
- ✅ İnteraktif ROI seçimi
- ✅ Kalibrasyon aracı (web üzerinde)
- ✅ Analiz ayarları (slider'lar ile)
- ✅ Gerçek zamanlı progress gösterimi
- ✅ Detaylı istatistikler ve görselleştirme
- ✅ Sonuçları indirme (JSON + Video)

## Kullanım

### Temel Kullanım

```bash
python main.py
```

Bu komut `veriseti_2.mp4` dosyasını analiz eder ve sonuçları `statistics.json` dosyasına kaydeder. ROI belirtilmediğinde, yoğunluk hesaplaması için tüm frame alanı kullanılır.

### ROI (Region of Interest) Tanımlama

Yoğunluk analizini sadece yol alanında yapmak için ROI tanımlayabilirsiniz:

```bash
# İnteraktif ROI seçici ile
python select_roi.py veriseti_2.mp4 roi_coordinates.json

# Seçilen ROI ile analiz
python main.py --video veriseti_2.mp4 --roi roi_coordinates.json

# Komut satırından ROI koordinatları ile
python main.py --video veriseti_2.mp4 --roi "100,200,800,200,900,600,50,600"
```

### Gelişmiş Kullanım

```bash
# Hızlı işleme için frame atlama ile
python main.py --video veriseti_2.mp4 --skip-frames 2 --output-video output.mp4

# Görüntüleme olmadan sadece analiz
python main.py --video veriseti_2.mp4 --no-display

# ROI ile tam özellikli analiz
python main.py --video veriseti_2.mp4 --model yolov8n.pt --confidence 0.25 --roi roi_coordinates.json --output-video output.mp4 --output-json results.json
```

### Hız ve Yön Analizi

Hız ve yön analizi **kalibrasyon olmadan da çalışır**! Kalibrasyon olmadan hız **piksel/saniye** cinsinden hesaplanır. Kalibrasyon ile **km/h** cinsinden gerçek dünya hızına dönüştürülebilir.

**Kalibrasyon Olmadan Kullanım (Önerilen - En Kolay):**
```bash
# Sadece tracking'i açın, kalibrasyon gerekmez!
python main.py --video veriseti_2.mp4

# Hız piksel/saniye (px/s) cinsinden gösterilir
# Yön tespiti ve araç takibi çalışır
# Hız limiti kontrolü yapılmaz (kalibrasyon gerekli)
```

**Kalibrasyon ile Kullanım (km/h için):**
Kalibrasyon, video'daki piksel mesafelerini gerçek dünya mesafelerine (metre) çevirmek için kullanılır.

#### Kalibrasyon Nedir?

Kalibrasyon, video'da ölçtüğünüz piksel mesafesini gerçek dünya mesafesine (metre) çevirmek için bir oran belirleme işlemidir.

**Örnek:** Video'da bir şerit genişliği 100 piksel ise ve gerçekte bu şerit 3.5 metre ise, kalibrasyon oranı: 100 piksel = 3.5 metre

#### Kalibrasyon Yapmanın 3 Yolu

##### Yöntem 1: İnteraktif Kalibrasyon Aracı (ÖNERİLEN - En Kolay)

En kolay yöntem, otomatik kalibrasyon aracını kullanmaktır:

```bash
# Kalibrasyon aracını çalıştır
python calibrate_speed.py veriseti_2.mp4

# Araç açıldığında:
# 1. Video'da bilinen bir mesafeyi seçin (örn: şerit genişliği)
# 2. İki noktaya tıklayarak bu mesafeyi ölçün
# 3. Gerçek mesafeyi metre cinsinden girin (örn: 3.5)
# 4. Kalibrasyon otomatik olarak kaydedilir
```

Kalibrasyon tamamlandıktan sonra:

```bash
# Kaydedilen kalibrasyon ile analiz yap
python main.py --video veriseti_2.mp4 --pixels-per-meter [araçtan_öğrendiğiniz_değer]
```

##### Yöntem 2: Manuel Referans Mesafe ile

Video'da bilinen bir mesafeyi manuel olarak ölçüp kullanabilirsiniz:

```bash
# Format: "piksel,metre"
# Örnek: 100 piksel = 3.5 metre (standart şerit genişliği)
python main.py --video veriseti_2.mp4 --reference-distance "100,3.5"
```

**Nasıl Yapılır:**
1. Video'yu bir görüntü düzenleyicide açın
2. Bilinen bir mesafeyi seçin (örn: şerit genişliği 3.5m, yol işareti arası 50m)
3. Bu mesafenin piksel cinsinden uzunluğunu ölçün
4. `--reference-distance "piksel,metre"` formatında kullanın

**Yaygın Referans Mesafeler:**
- Standart şerit genişliği: **3.5 metre**
- Yol işareti arası mesafe: **50 metre**
- Yol kenarı çizgisi uzunluğu: **3-6 metre**

##### Yöntem 3: ROI ve Şerit Bilgisi ile

ROI (yol alanı) genişliğini ve şerit sayısını biliyorsanız:

```bash
# Format: "genişlik_piksel,şerit_sayısı,şerit_genişliği_metre"
# Örnek: 800 piksel genişlik, 2 şerit, her şerit 3.5m
python main.py --video veriseti_2.mp4 --roi roi_coordinates.json --roi-width-lanes "800,2,3.5"
```

#### Hız Limitli Analiz

Kalibrasyon yapıldıktan sonra hız limiti belirleyebilirsiniz:

```bash
# 60 km/h hız limiti ile analiz
python main.py --video veriseti_2.mp4 --speed-limit 60 --pixels-per-meter 28.57
```

#### Tracking Olmadan Kullanım

Sadece araç sayımı yapmak istiyorsanız (hız/yön analizi olmadan):

```bash
python main.py --video veriseti_2.mp4 --no-tracking
```

#### Kalibrasyon Örnekleri

**Örnek 1: Şerit Genişliği ile**
```bash
# Video'da bir şerit genişliği 120 piksel, gerçekte 3.5 metre
python main.py --video veriseti_2.mp4 --reference-distance "120,3.5"
```

**Örnek 2: Direkt Oran ile**
```bash
# Eğer 1 metre = 28.57 piksel olduğunu biliyorsanız
python main.py --video veriseti_2.mp4 --pixels-per-meter 28.57
```

**Örnek 3: Tam Özellikli Analiz**
```bash
# ROI + Kalibrasyon + Hız Limiti + Çıktı Video
python main.py \
  --video veriseti_2.mp4 \
  --roi roi_coordinates.json \
  --reference-distance "100,3.5" \
  --speed-limit 60 \
  --output-video output.mp4 \
  --output-json results.json
```

### Parametreler

- `--video`: Analiz edilecek video dosyasının yolu (varsayılan: veriseti_2.mp4)
- `--model`: YOLO model dosyası (varsayılan: yolov8n.pt)
- `--confidence`: Tespit için güven eşiği (varsayılan: 0.25)
- `--roi`: ROI koordinatları - virgülle ayrılmış değerler (örn: "x1,y1,x2,y2,x3,y3,...") veya select_roi.py ile oluşturulan JSON dosyası yolu. Belirtilmezse tüm frame alanı kullanılır.
- `--output-video`: Görselleştirmeli çıktı video dosyası (opsiyonel)
- `--output-json`: İstatistiklerin kaydedileceği JSON dosyası (varsayılan: statistics.json)
- `--display`: İşlem sırasında videoyu göster (varsayılan: açık)
- `--no-display`: Görüntülemeyi kapat
- `--skip-frames`: İşlenecek frame'ler arasında atlanacak frame sayısı (0=hepsi, 1=her 2. frame, 2=her 3. frame, vb.) - Hızlandırma için kullanılır
- `--no-tracking`: Araç takibini kapat (hız ve yön analizi olmadan sadece sayım)
- `--pixels-per-meter`: Hız hesaplama için kalibrasyon: piksel/metre oranı
- `--reference-distance`: Kalibrasyon: "piksel,metre" formatında (örn: "100,3.5")
- `--roi-width-lanes`: Kalibrasyon: "genişlik_piksel,şerit_sayısı,şerit_genişliği_metre" (örn: "800,2,3.5")
- `--speed-limit`: Hız limiti (km/h) - ihlal tespiti için (varsayılan: 50.0)

## Proje Yapısı

```
traffic_analysis/
├── domain/                    # Domain Layer
│   ├── entities/             # İş mantığı varlıkları
│   │   ├── vehicle.py
│   │   ├── traffic_statistics.py
│   │   └── roi.py
│   └── use_cases/            # İş mantığı kullanım senaryoları
│       └── count_vehicles.py
├── application/               # Application Layer
│   ├── interfaces/           # Servis arayüzleri
│   │   ├── vehicle_detector.py
│   │   └── video_processor.py
│   └── services/             # İş mantığı servisleri
│       └── traffic_analyzer.py
├── infrastructure/            # Infrastructure Layer
│   ├── detection/            # Tespit implementasyonları
│   │   └── yolo_vehicle_detector.py
│   └── video/                # Video işleme implementasyonları
│       └── opencv_video_processor.py
├── presentation/              # Presentation Layer
│   ├── visualization/        # Görselleştirme
│   │   └── video_visualizer.py
│   └── reporting/            # Raporlama
│       └── statistics_reporter.py
├── main.py                   # Ana giriş noktası
├── select_roi.py             # ROI seçici araç
├── calibrate_speed.py         # Hız kalibrasyon aracı (interaktif)
├── requirements.txt          # Python bağımlılıkları
└── README.md                 # Bu dosya
```

## Clean Architecture

Proje Clean Architecture prensiplerine uygun olarak yapılandırılmıştır:

- **Domain Layer**: İş mantığı varlıkları ve kullanım senaryoları
- **Application Layer**: İş mantığı servisleri ve arayüzler
- **Infrastructure Layer**: Harici kütüphanelerin implementasyonları
- **Presentation Layer**: Kullanıcı arayüzü ve çıktı formatları

Bu yapı sayesinde:
- Kod daha test edilebilir
- Bağımlılıklar tersine çevrilmiştir (Dependency Inversion)
- Farklı implementasyonlar kolayca değiştirilebilir
- Kod daha modüler ve bakımı kolaydır

## Çıktı Formatı

Analiz sonuçları JSON formatında kaydedilir:

```json
{
  "total_vehicles": 1234,
  "max_vehicles_in_frame": 15,
  "average_vehicles_per_frame": 8.5,
  "peak_density": 0.0234,
  "vehicles_by_type": {
    "car": 800,
    "truck": 200,
    "bus": 50,
    "motorcycle": 184
  },
  "total_frames": 1000,
  "start_time": "2024-01-01T12:00:00",
  "end_time": "2024-01-01T12:05:00",
  "tracking": {
    "total_tracked_vehicles": 150,
    "average_speed_kmh": 45.2,
    "max_speed_kmh": 78.5,
    "total_speed_violations": 12,
    "vehicles_by_direction": {
      "up": 45,
      "down": 60,
      "left": 20,
      "right": 25
    },
    "speed_statistics_by_direction": {
      "up": {
        "average": 42.3,
        "max": 65.0,
        "min": 15.0,
        "count": 45
      },
      "down": {
        "average": 48.1,
        "max": 78.5,
        "min": 20.0,
        "count": 60
      }
    }
  }
}
```

## Gereksinimler

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- NumPy
- Pandas (raporlama için)

## Lisans

Bu proje eğitim amaçlıdır.

