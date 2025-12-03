# Traffic Analysis - Vehicle Counting and Density Analysis

Bu proje, video üzerinde araç sayımı ve yoğunluk analizi yapan bir Python uygulamasıdır. Clean Architecture prensiplerine uygun olarak geliştirilmiştir ve YOLOv8 ile OpenCV kullanmaktadır.

## Özellikler

- Araç tespiti (YOLOv8)
- Araç sayımı ve yoğunluk analizi
- ROI (Region of Interest) desteği - yol alanı tanımlama
- Detaylı istatistikler
- Video görselleştirme
- JSON formatında raporlama
- Clean Architecture mimarisi

## Kurulum

1. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

2. YOLOv8 modeli otomatik olarak indirilecektir (ilk çalıştırmada).

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
  "end_time": "2024-01-01T12:05:00"
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

