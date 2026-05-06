# Video Analyzer API

Bu proje, gelişmiş bir yapay zeka destekli Video Analiz API'si sunmaktadır. Yüklenen video dosyalarından anlam, bağlam ve ses dökümü çıkarmak için modern makine öğrenimi modellerini kullanır. Ses dökümü ve görsel anlama yeteneklerini birleştirerek video içeriği hakkında kapsamlı bilgiler (özet, etiketler ve niyet) üretir.

## Özellikler

- **Ses Dökümü (Transcription):** OpenAI'nin **Whisper** modelini kullanarak videodan sesi çıkarır ve metin segmentlerine dönüştürür.
- **Görsel Anlama (Visual Understanding):** Güçlü **Video-LLaVA-7B** modelini kullanarak video karelerini analiz eder.
- **Çok Modlu Bağlam (Multimodal Context):** Görüntü ve Dil modeline (VLM) videoda neler olduğunu ve neler konuşulduğunu tam olarak aktarabilmek için ses dökümünü görsel karelerle zaman damgalarına göre eşleştirir.
- **Yapılandırılmış Çıktı:** Videonun `summary` (özeti), ilgili `tags` (etiketleri) ve videonun `intent` (amacı/niyeti) bilgilerini içeren bir JSON yanıtı üretir.
- **GPU Uyumlu Konteyner:** Ağır makine öğrenimi iş yüklerinin GPU üzerinde verimli bir şekilde çalışmasını sağlamak için NVIDIA CUDA desteğiyle tamamen Dockerize edilmiştir.

## Gereksinimler

- CUDA destekli **NVIDIA GPU**
- **Docker** ve **Docker Compose**
- Konteynerlere GPU erişimi verebilmek için **NVIDIA Container Toolkit** (nvidia-docker)

## Kurulum ve Çalıştırma

1. **Projeyi indirin:**
   ```bash
   git clone https://github.com/emrecorbacii/video-analyzer.git
   cd video-analyzer
   ```

2. **Docker Compose ile başlatın:**
   ```bash
   docker-compose up --build
   ```
   *Not: İlk kurulumda büyük model ağırlıkları (Whisper ve Video-LLaVA) indirileceği ve PyTorch ile CUDA bağımlılıkları kurulacağı için bu işlem biraz zaman alabilir.*

3. **API Hazır:**
   Modeller belleğe yüklendiğinde FastAPI sunucusu `http://localhost:8000` adresinde çalışmaya başlayacaktır.

## Kullanım

Uygulama, video analizi için tek bir uç nokta (endpoint) sunar.

**Uç Nokta:** `POST /analyze`

Bir videoyu analiz etmek için `curl` kullanarak şu şekilde istek atabilirsiniz:

```bash
curl -X POST http://localhost:8000/analyze \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/bilgisayardaki/yol/video.mp4"
```

### Örnek Yanıt
API, modelin ürettiği veriyi şu formatta döndürür:
```json
{
  "result": "{\n  \"summary\": \"Videonun genel özeti burada yer alır.\",\n  \"tags\": [\"etiket1\", \"etiket2\"],\n  \"intent\": \"eğitim\"\n}"
}
```

## Kullanılan Teknolojiler
- **FastAPI:** API için yüksek performanslı web framework'ü.
- **PyTorch:** Derin öğrenme altyapısı.
- **OpenAI Whisper:** Sesi metne dönüştürme (Speech-to-text).
- **Video-LLaVA:** Videoları anlama yeteneğine sahip VLM (Vision-Language Model).
- **FFmpeg & OpenCV:** Video ve ses işleme kütüphaneleri.
- **Docker:** Konteynerleştirme.