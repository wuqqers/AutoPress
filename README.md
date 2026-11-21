# AutoPress

## Tanılama ve ayrı doğrulama verisi

Sağlanan `AutoPress_fixes` paketindeki tanılama, neden kodları, aykırı eğitim
örneği kontrolü ve değerlendirme fikirleri güncel sinir ağı/öğrenme koduna
uyarlandı. Dosyalar topluca değiştirilmedi. CMN seçeneği, gerçek konuşmada
iyileşme gösterilmediği için etkinleştirilmedi.

Deneme sekmesindeki **Doğrulama özeti**, doğru kabulü, yanlış komutu, reddedilen
doğru adayı, başka adayın öne çıktığı kaçırmaları ve ses kalitesi hatalarını
ayırır. Komut dışı konuşmayı ölçmek için hedefe `-` yazın. Etiketsiz canlı
dinleme doğruluk veya yanlış tetikleme oranı olarak sayılmaz. Özet yalnızca
bu oturumdaki, en fazla 200 kişisel tanıma kaydını kapsar; Vosk ve deneysel
cümle içinde arama için aynı kapsamda bir doğrulama raporu değildir.

Ses kuyruğunda bekleme artık mikrofon callback varış zamanıyla ölçülür.
Özellik/eşleştirme hesaplama süresi, sessizlik beklemesi, konuşma sonundan
karara ve Windows tuş çağrısının dönüşüne kadar süreler ayrıdır. Mikrofon
donanımı/sürücüsü içinde bekleme ve oyunun tuşu işleme süresi ölçülmez.
Tuş gönderimi, odak nedeniyle engelleme, test modu ve gönderme hataları da
tanınan söyleyişin kaydına bağlanır.

**Doğrulama özelliklerini RAM'de tut** kutusu isteğe bağlıdır ve varsayılan
kapalıdır. Açıp yeni söyleyişlerle test yaparken öğrenme durur. Dışa aktarma
düğmesi, yalnız sayısal özellikleri ve hedef etiketlerini seçtiğiniz JSON'a
yazar; ham ses yazmaz. Sayısal özellikler de kişiseldir. Eğitimdeki örnekleri
tekrar test verisi gibi kullanmayın. Hedefi değiştirirseniz testi yeniden başlatın.

Kaydedilmiş yeni söyleyişler üç karar moduyla tekrar puanlanabilir:

```powershell
.\.venv-tk\Scripts\python.exe -m autopress.evaluate "$env:LOCALAPPDATA\AutoPress\profiles.json" "LOL" autopress_validation.json
```

Bu komut yalnız MFCC v2 veya sinir ağı sınıflandırıcısını yeniden çalıştırır;
VAD, ek akustik doğrulama ve tuş gönderimini taklit etmez. Bunların sonucu
uygulama özetindedir. Eğitim veya ek öğrenme değişmişse eski doğrulama kümesini
güncel modelle karıştırmamak için değerlendirme hata verir.

Öğretmede canlı dinlemeyle aynı sessizlik ayarı kullanılır. Diğer dört örnekten
süre veya spektrum bakımından çok farklı bir örnek bulunursa yalnız o örnek
yeniden istenir. Üç ek denemeden sonra hâlâ tutarsızsa eğitim durur; bilinen
aykırı örnek zorla kabul edilmez. Bu sezgisel bir kalite kontrolüdür; doğru
telaffuzun yerini tutmaz.

## Birbirine benzeyen harflerde ek ses doğrulaması

**Benzer seslerde ek doğrulama** varsayılan açıktır. Sinir ağının kabul ettiği
komut ayrıca bağımsız MFCC/DTW kontrolüyle karşılaştırılır. İki kontrol aynı
komutu yeterli ayrımla desteklemezse tuş gönderilmez ve söyleyiş otomatik
öğrenilmez. Deneme sonucunda **ses kontrolleri uyuşmadı** nedeni gösterilir.
Bu, yanlış tuşu azaltmayı amaçlar; daha fazla kararsız/kaçırılan komut olabilir.

Yeni sinir ağı eğitiminde aynı ham ses RAM'de iki ayrı temsil için işlenir;
`phonetic_samples` sayısal alanı da örneklerle saklanır. Ham ses yine yazılmaz.
Eski sinir ağı kayıtlarında bu alan yoksa, profilin tüm komutları için mevcut
sürüm 2 MFCC örnekleri kullanılabilir. Herhangi bir komutun karşılaştırma kaydı
eksikse doğrulama açılmaz; dinleme durumunda **kayıt yok** bilgisi gösterilir.
Bu durumda komutları yeni sürümle yeniden öğretin. Ayar değişikliğinden sonra
dinlemeyi yeniden başlatın. İki kontrolün anlaşması kesin doğruluk garantisi
değildir; gerçek telaffuzunuzla Deneme bölümünde doğrulayın.

## Manuel ve otomatik kişisel öğrenme

Sinir ağı motoru artık kullanım sırasında kişisel sınıflandırma başlığını
güncelleyebilir. Önceden eğitilmiş CNN ve ilk beş kayıt sabit kalır. Kullanımın
artması tek başına işlem süresini düşürmez; amaç tanımayı kişiye uyarlamaktır.

**Manuel:** Canlı dinleme veya Deneme sonrasında, Deneme sekmesindeki geri
bildirim bölümünden son 20 söyleyişten birini seçin. Doğru sonuçta **Doğru —
Öğren**, hatalı veya reddedilmiş sonuçta doğru komutu seçip **Seçilen komut
olarak öğret** kullanın. Söyleyişler inceleme için yalnız RAM'de beş dakika
tutulur; ham ses tutulmaz. Onaylanan 96 boyutlu özellikler profile kaydedilir.
Komut olmayan sonuçta **Komut değil / Öğrenme** ile o söyleyişin ek öğrenmesini
kaldırabilirsiniz. Manuel düzeltme aynı söyleyişin otomatik etiketini değiştirir.

**Otomatik:** Ayarlar'daki otomatik öğrenme seçeneği varsayılan açıktır. İlk
eğitimden gelen sabit model ve uyarlanan model aynı komutta güçlü biçimde
anlaşmalı; üç farklı söyleyiş destek vermelidir. Bir komut için en sık 10 saniyede
bir ekleme yapılır. Sistem doğruluğu veya oyundaki niyeti kesin olarak bilemez;
bu, gerçek etiket yerine yüksek güvenli tahminle öğrenmedir. Kararsız ve
reddedilen sonuçlar otomatik öğrenilmez. Deneme hedefiyle çelişen sonuç da
otomatik öğrenilmez. Manuel olarak düzeltilen/reddedilen tahmin etiketinin
otomatik öğrenmesi, ek öğrenme sıfırlanana veya yeniden eğitim yapılana kadar
duraklatılır. Diğer komutlarda otomatik öğrenme devam edebilir.

Her komut için en fazla 8 manuel ve 3 otomatik ek örnek tutulur; eski ek örnekler
sırayla yenilenir. İlk kayıtlar değiştirilmez. Yeni başlık ilk kayıtların önceden
doğru tanınanlarını bozmamalıdır; aksi durumda güncelleme uygulanmaz. Bu kontrol
yeni konuşmalar için doğruluk garantisi değildir. Ek veriler
`adaptive_learning`, açık/kapalı tercihi `auto_learning` alanında saklanır.
Yeni başlangıç eğitimi eski ek öğrenmeyi geçersiz kılar.

Kaydedilen değişiklik dinleme sürerken, tuş gönderimi tamamlandıktan sonraki
ses bloklarında uygulanır. Ayarlar'dan son beş değişikliği bu oturumda geri
alabilir veya **ilk kayıtları koruyarak bütün ek öğrenmeyi silebilirsiniz**.
Ekran kalıcı manuel/otomatik örnek sayılarını gösterir. Eski MFCC ve Vosk
motorlarında bu öğrenme uygulanmaz.

## Yeni: beş örnekle kişisel sinir ağı

Varsayılan **Kişisel sinir ağı (yeni)**, Google'ın önceden eğitilmiş konuşma CNN'ini
ONNX Runtime ile CPU'da çalıştırır. openWakeWord v0.5.1'den gelen gerçek model
ağırlıkları pakete dahildir ve SHA-256 ile kontrol edilir. Model sessizce MFCC'ye
dönmez; eksik/bozuk modelde açıklamalı hata verir. Lisans ve kaynaklar için
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) dosyasına bakın.

Sinir ağı her söyleyişten 96 boyutlu temsil çıkarır. Beş kayıt/komut ile yerelde
düzenlileştirilmiş bir dense sınıflandırma katmanı eğitilir. Rakip komutlar,
örneklere uzaklık ve karar farkı birlikte kontrol edilir. Bunlar kullanıcıya özel
few-shot öğrenmedir; omurga yeniden eğitilmez. Uzun kayıtlar örtüşen pencerelerde
işlenir ve havuzlanır. Kısa kayıtlar RAM'de doldurulur; modelin bağlamı dolsun diye
fazladan gerçek ses beklenmez. Normal konuşma sonu beklemesi yine uygulanır.

1. Yöntemden **Kişisel sinir ağı (yeni)** seçin.
2. Her komut için beş yeni örnek söyleyin. Aynı harfi aynı telaffuzla öğretin.
3. Liste yeni eğitimde tamamlanan ve kalan komutları gösterir. Bütün komutlar
   bitince sinir ağı kayıtları etkinleşir.
4. Önce **Deneme** bölümünde eğitimde kullanmadığınız söyleyişleri sınayın.
   Seçici/Dengeli/Atak ayarı bu motor için de geçerlidir.

Eski MFCC kayıtlarından sinir ağı temsili üretilemez; yeniden öğretmek gerekir.
Eski kayıtlar `audio_templates`, yeni kayıtlar `neural_templates` alanında tutulur.
Yarım eğitim `pending_neural_templates` alanına kaydedilir. Eski motor menüden
seçilebilir. Hiçbir kullanıcı sesi sunucuya gönderilmez veya dosyaya yazılmaz;
yalnız sayısal sinir ağı temsilleri açık kayıt onayıyla saklanır.

Bu, Türkçe tek harflerde kesin başarı garantisi değildir. Ön eğitim Türkçe harf
ayrımına özel değildir. Gerçek söyleyişler, komutsuz konuşma, mikrofon ve oyun
yükü altında yanlış kabul/ret ve toplam gecikme ayrıca değerlendirilmelidir.
Sentetik testler doğal konuşma doğruluğu sayılmaz. Önceki cümle içinde arama
ve MFCC erken karar seçenekleri sinir ağı motorunda kullanılmaz.

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
kuyruk beklemesi ayrı ölçülmez. Her ayrı söyleyiş tek eylem üretir.

## Veriler

Profil dosyası: `%LOCALAPPDATA%\AutoPress\profiles.json`.

Ham ses dosyaya yazılmaz; işleme sırasında sınırlı RAM tamponlarında tutulur.
Kişisel örüntüler sayısal ses özellikleri olarak kaydedilir. Bunlar ses karakteristiği
barındırabileceğinden profil dosyanızı paylaşırken dikkat edin. **Bu profilin ses
özelliklerini sil** düğmesiyle kaldırılabilirler. Profiller, modeller ve ses dosyaları
Git deposuna dahil edilmez.

## Geliştirme

### Tek dosyalı Windows EXE

```powershell
.\.venv-tk\Scripts\python.exe -m pip install -e ".[build]"
.\.venv-tk\Scripts\python.exe -m PyInstaller --noconfirm AutoPress.spec
```

Çıktı `dist/AutoPress.exe` dosyasıdır. Python veya BAT dosyası olmadan çalışır;
ilk açılışta paket geçici klasöre açıldığı için başlangıç biraz sürebilir.
Kişisel profiller ve Vosk modeli EXE'ye gömülmez. Mevcut kullanıcı verileri
`%LOCALAPPDATA%\AutoPress` konumundan okunur. Başka bilgisayarda kişisel ses
örnekleri oluşturulmalı; Vosk kullanılacaksa model uygulama içinden indirilmelidir.

Paketin arayüz ve bağımlılık kontrolü (mikrofon kaydı ve tuş gönderimi yapmaz):

```powershell
Start-Process -FilePath .\dist\AutoPress.exe -ArgumentList '--smoke-test', 'build/smoke-test.json' -Wait
Get-Content build/smoke-test.json
```

### Testler

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

## Kısa komutlar için yeni kişisel eğitim

### Cümle içinde kişisel komut arama

Yeni eğitimle Ayarlar'daki **Cümle içinde kişisel komut ara** seçeneği kullanılabilir.
Varsayılan kapalıdır; önce tek komutlarla Deneme'de doğrulama yapılmalıdır.
Sürüm 1 kayıtları konuşma sonunu bekleyen eski yöntemi kullanır.
Sürüm 2 örnekler, akış içindeki herhangi bir başlangıçtan başlayabilen alt dizi
DTW ile aranır. Cümlenin tamamlanması gerekmez; hem kelime hem çok kelimeli kayıt
aranabilir. Her 10 ms ses karesi bir kez işlenir; kararlar 60 ms aralıklarla
kontrol edilir. İki kayıtla eşleşme ve komutlar arası fark gerekir. Çok belirgin
eşleşme doğrudan, diğer eşleşmeler iki kontrol sonunda kabul edilebilir.

Bu yaklaşım kişisel akustik örneği arar; cümlenin anlamını veya komut verme niyetini
yorumlamaz. Sohbet sırasında geçen kayıtlı sözcük de tetikleyebilir. Önce Deneme
modunda hem komutlarla hem ilgisiz konuşmayla sınayın. Gerçek konuşmada doğruluk
iyileşmesi henüz doğrulanmamıştır; deneysel özellik kapatılarak önceki yönteme
dönülebilir. Bu özellik yeni bir sinir ağı modeli içermez.

Akış modunda sessizlik beklemesi ve eski erken karar ayarı kullanılmaz. Ekrandaki
akış süresi yalnız son ses bloğunun işlenme süresidir; konuşma sonundan tuşa kadar
ölçüm veya toplam tepki süresi değildir.

Yeni örnekler 24 katsayılı sürüm 2 olarak alınır. Bütün profil komutlarını beşer
kez yeniden öğretin; tamamlanana kadar eski kayıtlar kullanılmaya devam eder.
Ortak normalizasyon ve diğer komutlarla karşılaştırma dinleme başında hesaplanır.
Karar, en yakın iki kayıtla desteklenir. Eğitim sonunda karışan komutlar belirtilir.

Ayarlar'da Seçici, Dengeli veya Atak seçilebilir. Atak daha fazla sesi kabul eder;
yanlış tuş riski yükselir. Değişiklikler yeni dinlemede uygulanır. Deneme kartları
hedef başına doğru, reddedilen ve yanlış sonuçları sayar. Her komutu en az beş
kez, öğretirken kullandığınız telaffuzla deneyin. Gerçek doğruluk değerlendirmesi
 için komut başına 30 yeni söyleyiş ve komutsuz arka plan denemesi önerilir.

Erken karar varsayılan kapalıdır. Bu oturumda beş doğru ve sıfır hatalı veya
reddedilen denemesi olan komutlar için Ayarlar'dan etkinleştirilebilir. İki güçlü
sonuçta, en az 60 ms sessizlikten sonra karar verebilir; diğer durumlarda normal
sessizlik ayarını bekler. Yeniden eğitimde ve karar modu değiştiğinde kapanır.
Bu küçük deneme doğruluk garantisi değildir; normal konuşmanızla sınayın.

Kısa komutlarda ses başlangıcını korumak için 100 ms ön ses tutulur. Ses 10 ms
parçalarla işlenir. Aynı komutun yeni bir söyleyişi için eski bir saniyelik engel
kaldırılmıştır. Ana gecikme göstergesinin ayrıntıları karar puanlarını ve kişisel
normal karar için konuşma sonundan tuş gönderimine kadar süreyi de gösterir.
