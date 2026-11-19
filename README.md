# AutoPress

Türkçe sesli komutları klavye tuşlarına dönüştüren Windows masaüstü uygulaması.
Komutları ve kişisel ses örneklerini oyun profillerine göre ayırır. Tanıma yerel
çalışır; konuşma bir sunucuya gönderilmez.

## Özellikler

- **Kişisel ses örüntüsü:** Her komut için beş örnekle MFCC/DTW eşleştirmesi.
- **Vosk:** Alternatif olarak yerel Türkçe konuşma tanıma.
- **Tuş yakalama:** Tuşa basarak eşleme; kısa basış veya basılı tutma.
- **Oyun profilleri:** Ayrı komutlar, ses özellikleri, pencere filtresi ve bekleme süresi.
- **Deneme modu:** Tuş göndermeden tanıma sonucunu ve süresini görme.
- **Sistem tepsisi:** Pencere kapalıyken dinlemeye devam etme.

Varsayılan yöntem kişisel ses örüntüsüdür. Tuşlar varsayılan olarak Windows
`SendInput` üzerinden tarama koduyla gönderilir.

## Kurulum

Windows 10/11, Python 3.11 ve bir mikrofon gerekir. Proje klasöründe PowerShell açın:

```powershell
py -3.11 -m venv .venv-tk
.\.venv-tk\Scripts\python.exe -m pip install --upgrade pip
.\.venv-tk\Scripts\python.exe -m pip install -e .
.\run_autopress.bat
```

Sanal ortamı etkinleştirmek gerekmez. Sonraki açılışlarda `run_autopress.bat`
dosyasına çift tıklayabilirsiniz. `.venv-tk` klasör adı başlatıcılarla uyumluluk için
korunmuştur; arayüz PySide6 kullanır.

Vosk yöntemini veya sesle komut ekleme yardımcısını kullanacaksanız **Ayarlar →
Türkçe modeli indir** düğmesiyle modeli indirin. Kişisel örüntü eşleştirmesi Vosk
modeli gerektirmez. Model indirme internet bağlantısı ister.

## Kullanım

1. Oyun profilini ve mikrofonu seçin.
2. **Komutlar** bölümünde komut sözcüğünü girin. **Tuşu yakala** ile fiziksel tuşa
   basın, kısa basış veya basılı tutma davranışını seçip kaydedin.
3. Kişisel yöntem için komutu seçin ve **5 kez ses örneği kaydet** akışını tamamlayın.
   Sayısal ses özellikleri onayınızla profile otomatik kaydedilir.
4. **Deneme** bölümünde komutu söyleyin. Bu mod klavye tuşu göndermez.
5. **Ayarlar** bölümünden hedef pencereyi seçin ve dinlemeyi başlatın.

Pencere başlığı boşsa komutlar tüm ön plan pencerelerinde çalışır. Bir metin
belirtilirse pencere filtresi açıkken yalnızca başlığında bu metin olan pencereye
komut gönderilir. **Pencereyi seç** düğmesi hedef pencereye geçmek için sekiz saniye verir.

Basılı tutma modunda aynı komutu tekrar söylemek tuşu bırakır. **Tüm tuşları bırak**
düğmesiyle de bırakabilirsiniz. Pencereyi kapatmak uygulamayı tepsiye küçültür;
tamamen çıkmak için tepsi menüsündeki çıkış seçeneğini kullanın.

## Gecikme ayarı

**Ayarlar → Kişisel örüntü · Sessizlik beklemesi** alanından 100–600 ms seçilebilir.
Başlangıç değeri 150 ms'dir. Ayar oyun profiline otomatik kaydedilir; yeni değeri
uygulamak için dinlemeyi durdurup yeniden başlatın. Daha kısa bekleme tepkiyi
hızlandırabilir, ancak kelime içindeki duraklamalarda sözü erken kesebilir.

Kişisel yöntemde gecikme sayısının üzerine gelerek özellik çıkarma, eşleştirme ve
konuşma sonunu bekleme sürelerini görebilirsiniz. Ana süre konuşma başlangıcından
tanıma kararına kadardır; oyunun tuşu işleme süresini kapsamaz. Sürücü tamponu ve
kuyruk beklemesi ayrı ölçülmez. Aynı komutun tekrar gönderilmesinde 1000 ms koruma vardır.

## Veriler

Profil dosyası: `%LOCALAPPDATA%\AutoPress\profiles.json`.

Ham ses dosyaya yazılmaz; işleme sırasında sınırlı RAM tamponlarında tutulur.
Kişisel örüntüler sayısal ses özellikleri olarak kaydedilir. Bunlar ses karakteristiği
barındırabileceğinden profil dosyanızı paylaşırken dikkat edin. **Bu profilin ses
özelliklerini sil** düğmesiyle kaldırılabilirler. Profiller, modeller ve ses dosyaları
Git deposuna dahil edilmez.

## Geliştirme

```powershell
.\.venv-tk\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv-tk\Scripts\python.exe -m pytest -p no:unittest -q
.\.venv-tk\Scripts\python.exe tools/benchmark_acoustic.py
```

Testler gerçek mikrofon kaydı veya klavye girdisi gerektirmez. Performans betiği
sentetik örüntülerle seri ve toplu eşleştirmeyi karşılaştırır; uçtan uca oyun
gecikmesini ölçmez.

| Dosya | Görev |
| --- | --- |
| `src/autopress/app_tk.py` | PySide6 arayüzü |
| `src/autopress/recognition.py` | Ses akışı ve tanıma worker'ları |
| `src/autopress/acoustic.py` | MFCC çıkarma ve DTW eşleştirme |
| `src/autopress/keys.py` | Windows tuş gönderimi |
| `src/autopress/profiles.py` | Profil doğrulama ve kayıt |

## Sınırlamalar

Benzer sözcükler, arka plan gürültüsü ve mikrofon seçimi tanımayı etkiler. Gerçek
komutlarınızı Deneme bölümünde kontrol edin. Bazı oyunlar sentetik klavye girdisini
kabul etmeyebilir; tüm oyunlarla uyumluluk garantisi yoktur.
