# Penjelasan arsitektur ClockwiseCNN

## Tujuan model

`ClockwiseCNN` adalah model klasifikasi biner untuk membaca urutan label A, B, dan C pada gambar kurva berwarna. Kelas `+` berarti urutan clockwise adalah `ABC`, sedangkan kelas `-` berarti `ACB`.

Model tidak menerima aturan clockwise dalam bentuk rumus. Model menerima pixel gambar dan mempelajari pola visual yang berhubungan dengan posisi label, bentuk kurva, dan hubungan spasial di antaranya.

## Dari file gambar ke tensor

Setiap file PNG asli berukuran 400 × 400 pixel. Sebelum masuk ke CNN, `train.py` melakukan langkah berikut.

1. Gambar diubah menjadi greyscale. Warna kurva tidak dipakai sebagai fitur utama karena warna berubah antar gambar, sedangkan kelas ditentukan oleh urutan label.
2. Gambar di-resize menjadi 128 × 128 pixel.
3. Nilai pixel diubah dari rentang 0 sampai 255 menjadi tensor dengan rentang 0 sampai 1.
4. Tensor dinormalisasi menggunakan mean dan standard deviation yang dihitung dari folder `training/` saja.

Saat training, gambar juga dapat diputar maksimal 8 derajat, digeser maksimal 4 persen, dan di-scale dari 0.95 sampai 1.05. Horizontal dan vertical flip tidak dipakai karena refleksi akan membalik clockwise menjadi counter-clockwise dan dapat mengubah label yang seharusnya.

Setelah preprocessing, satu batch masuk ke model dengan bentuk:

```text
[batch_size, channel, height, width]
[B, 1, 128, 128]
```

`B` adalah jumlah gambar dalam satu batch. Nilai default-nya adalah 16.

## Apa yang dilakukan __init__

Bagian `__init__` membangun daftar layer dan parameter yang akan dipelajari. Pada tahap ini gambar belum diproses. PyTorch hanya mendaftarkan convolution kernel, parameter BatchNorm, bobot Linear, dan bias ke dalam objek `ClockwiseCNN`.

Pemrosesan gambar baru terjadi saat `forward(images)` dipanggil. `forward` mengirim tensor gambar melalui `self.features`, lalu `self.classifier`.

```text
images
  │
  ▼
self.features
  │
  ▼
self.classifier
  │
  ▼
logit tunggal
```

Model memiliki 154,849 parameter yang dapat dipelajari. Parameter ini diperbarui oleh AdamW melalui backpropagation saat training.

## Bagian self.features

`self.features` adalah extractor fitur. Tugasnya mengubah pixel mentah menjadi representasi yang lebih berguna untuk membaca hubungan spasial.

| Tahap | Layer | Bentuk output | Peran pada gambar |
|---|---|---:|---|
| Input | Greyscale image | `[B, 1, 128, 128]` | Satu channel berisi struktur kurva, lingkaran, dan huruf. |
| Block 1 | Conv2d 1→16, BatchNorm, ReLU | `[B, 16, 128, 128]` | Mencari fitur kecil seperti tepi garis kurva, outline lingkaran label, dan stroke huruf. |
| Block 1 | MaxPool2d(2) | `[B, 16, 64, 64]` | Mengurangi resolusi setengah sambil mempertahankan fitur terkuat di tiap area 2 × 2. |
| Block 2 | Conv2d 16→32, BatchNorm, ReLU | `[B, 32, 64, 64]` | Menggabungkan fitur kecil menjadi pola lokal, misalnya bagian label yang dekat dengan bagian tertentu dari kurva. |
| Block 2 | MaxPool2d(2) | `[B, 32, 32, 32]` | Mengurangi resolusi lagi agar layer berikutnya melihat area gambar yang lebih luas. |
| Block 3 | Conv2d 32→64, BatchNorm, ReLU | `[B, 64, 32, 32]` | Membentuk fitur tingkat lebih tinggi tentang hubungan posisi antarlokasi pada gambar. |
| Spatial pooling | AdaptiveAvgPool2d((4, 4)) | `[B, 64, 4, 4]` | Merangkum gambar menjadi grid 4 × 4 sambil mempertahankan informasi kasar tentang posisi kiri, kanan, atas, dan bawah. |

### Conv2d

Semua convolution memakai kernel 3 × 3 dan `padding=1`. Karena padding ini, convolution tidak langsung mengubah tinggi dan lebar feature map. Contohnya, input `[B, 1, 128, 128]` menjadi `[B, 16, 128, 128]` setelah convolution pertama.

Setiap kernel bergerak di seluruh gambar. Kernel pertama dapat belajar mendeteksi garis lengkung atau stroke hitam pada huruf. Layer berikutnya menerima banyak feature map dari layer sebelumnya, jadi ia dapat belajar pola yang lebih kompleks daripada garis tunggal.

Jumlah channel meningkat dari 1 menjadi 16, 32, lalu 64. Ini memberi model lebih banyak jenis fitur untuk direpresentasikan saat informasi visual menjadi lebih kompleks.

### BatchNorm2d

`BatchNorm2d` menormalkan nilai feature map untuk setiap channel selama training. Nilai aktivasi menjadi lebih stabil, sehingga optimizer lebih mudah memperbarui bobot tanpa terlalu sensitif terhadap skala aktivasi dari layer sebelumnya.

### ReLU

`ReLU()` mengubah nilai negatif menjadi nol dan membiarkan nilai positif lewat. Tanpa aktivasi non-linear, beberapa convolution dan Linear layer hanya setara dengan satu transformasi linear besar. ReLU membuat model dapat mempelajari pola visual yang tidak linear.

### MaxPool2d

`MaxPool2d(2)` mengambil nilai terbesar pada setiap area 2 × 2. Ukuran feature map menjadi setengah, dari 128 × 128 ke 64 × 64, lalu ke 32 × 32. Ini membuat receptive field layer berikutnya mencakup area gambar yang lebih luas dan mengurangi jumlah komputasi.

### AdaptiveAvgPool2d((4, 4))

Layer ini mengubah setiap feature map berukuran 32 × 32 menjadi 4 × 4. Berbeda dari global average pooling 1 × 1, grid 4 × 4 masih menyimpan lokasi kasar fitur.

Hal ini penting untuk tugas ini. Jika model hanya menyimpan satu rata-rata untuk seluruh gambar, model dapat mengetahui bahwa ada kurva dan ada tiga label, tetapi kehilangan petunjuk apakah B muncul sebelum atau sesudah C saat mengelilingi gambar. Dengan grid 4 × 4, classifier masih dapat membedakan bukti fitur yang muncul di bagian gambar yang berbeda.

## Bagian self.classifier

Output dari `self.features` memiliki bentuk `[B, 64, 4, 4]`. Bagian classifier mengubahnya menjadi satu nilai untuk setiap gambar.

| Layer | Bentuk output | Fungsi |
|---|---:|---|
| `Flatten()` | `[B, 1024]` | Mengubah 64 × 4 × 4 fitur menjadi satu vektor. |
| `Linear(1024, 128)` | `[B, 128]` | Menggabungkan fitur dari berbagai area grid untuk mencari bukti urutan `ABC` atau `ACB`. |
| `ReLU()` | `[B, 128]` | Memberikan non-linearitas pada kombinasi fitur. |
| `Dropout(0.4)` | `[B, 128]` saat training | Menonaktifkan sekitar 40 persen aktivasi secara acak saat training untuk mengurangi ketergantungan pada fitur tertentu. Saat validation dan inference, Dropout dimatikan. |
| `Linear(128, 1)` | `[B, 1]` | Menghasilkan satu logit untuk klasifikasi biner. |

`Flatten()` menghasilkan 1024 nilai karena `64 × 4 × 4 = 1024`. Linear pertama tidak melihat pixel mentah lagi. Ia melihat fitur hasil convolution beserta lokasi kasarnya.

## Logit, probability, dan label akhir

Layer terakhir menghasilkan logit, bukan probability.

```text
logit < 0   cenderung kelas -
logit = 0   probability + = 0.5
logit > 0   cenderung kelas +
```

Saat training, `BCEWithLogitsLoss` menerima logit langsung dan membandingkannya dengan label target, yaitu 1 untuk `+` dan 0 untuk `-`. Loss ini sudah menghitung sigmoid secara stabil, sehingga model tidak memakai `Sigmoid()` sebagai layer eksplisit di `model.py`.

Saat inference, `inference.py` menghitung sigmoid dari logit:

```python
positive_probability = torch.sigmoid(model(batch).squeeze()).item()
```

Jika probability `+` minimal 0.5, prediksi adalah `+ (clockwise ABC)`. Jika kurang dari 0.5, prediksi adalah `- (clockwise ACB)`. Confidence untuk kelas `-` dihitung sebagai `1 - positive_probability`.

## Ringkasan aliran satu gambar

```text
PNG 400 × 400
  → greyscale
  → resize 128 × 128
  → normalize
  → Conv 1→16 + BatchNorm + ReLU + MaxPool
  → Conv 16→32 + BatchNorm + ReLU + MaxPool
  → Conv 32→64 + BatchNorm + ReLU
  → adaptive average pooling ke grid 4 × 4
  → flatten 1024 fitur
  → Linear 1024→128 + ReLU + Dropout
  → Linear 128→1 logit
  → sigmoid
  → probability dan kelas + atau -
```

Arsitektur ini cukup kecil untuk dataset 140 gambar training, tetapi tetap memiliki informasi posisi yang dibutuhkan untuk tugas clockwise-order. Hasil test tetap perlu dibaca bersama confusion matrix dan contoh error, terutama ketika B dan C berada sangat dekat.
