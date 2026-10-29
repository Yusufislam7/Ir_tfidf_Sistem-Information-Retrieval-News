# Sistem Information Retrieval Berita

Proyek ini merupakan tugas data mining untuk mencari artikel berita berdasarkan kata kunci dan membandingkan dua metode pemeringkatan dokumen: **TF-IDF dengan cosine similarity** dan **BM25**. Hasil pencarian menampilkan artikel paling relevan, ringkasan ekstraktif, serta metrik evaluasi sederhana.

Antarmuka utama proyek menggunakan Streamlit. Tersedia juga skrip terminal `ir_tfidf.py` untuk menjalankan pencarian langsung dari dataset TSV.

## Fitur

- Preprocessing teks berbahasa Inggris: mengubah huruf menjadi lowercase, menghapus karakter selain huruf, menghapus stopword, dan melakukan stemming dengan Porter Stemmer.
- Inverted index untuk menemukan dokumen kandidat pada skrip terminal.
- Pemeringkatan hasil menggunakan TF-IDF + cosine similarity dan BM25.
- Menampilkan hingga lima hasil untuk setiap metode.
- Ringkasan ekstraktif dua kalimat menggunakan skor TF-IDF per kalimat.
- Evaluasi Precision@5 dan Mean Reciprocal Rank (MRR) berdasarkan kategori dominan hasil.
- Antarmuka web untuk melihat kata kunci utama dan isi lengkap artikel.

## Struktur proyek

```text
.
├── app.py                       # Aplikasi Streamlit
├── ir_tfidf.py                  # Pencarian interaktif melalui terminal
├── split_bbc_csv.py             # Memecah dataset TSV menjadi file teks
├── bbc-news-data.csv            # Dataset berita berformat TSV
├── data_berita/                 # Kumpulan artikel teks untuk aplikasi Streamlit
├── Dokumentasi_Proyek_Kelompok_3.pdf
└── datminlaporan_final.docx
```

## Persyaratan

- Python 3.9 atau lebih baru
- Koneksi internet saat pertama kali dijalankan untuk mengunduh NLTK English stopwords

Pasang pustaka yang digunakan:

```bash
pip install streamlit pandas numpy nltk scikit-learn rank-bm25
```

## Menjalankan aplikasi web

Dari direktori proyek, jalankan:

```bash
streamlit run app.py
```

Streamlit akan menampilkan URL lokal di terminal (biasanya `http://localhost:8501`). Masukkan query berbahasa Inggris, misalnya `apple technology` atau `football match`, untuk membandingkan hasil TF-IDF dan BM25.

Aplikasi membaca file `.txt` dari direktori `data_berita/` relatif terhadap direktori kerja saat perintah dijalankan. Setiap file diperlakukan sebagai satu artikel. Baris pertama artikel digunakan sebagai judul, sedangkan isi file lengkap digunakan untuk pencarian dan ringkasan.

## Menjalankan pencarian terminal

```bash
python ir_tfidf.py
```

Skrip membaca `bbc-news-data.csv` sebagai TSV, lalu meminta kata kunci pencarian di terminal. Judul diberi bobot dua kali dengan cara mengulang judul sebelum isi artikel. Skrip menampilkan maksimal lima hasil TF-IDF dan lima hasil BM25, beserta ringkasan dan metrik evaluasinya.

## Membuat file artikel dari dataset

Untuk membuat atau membangun ulang file teks dari `bbc-news-data.csv`, jalankan:

```bash
python split_bbc_csv.py
```

Skrip menulis artikel ke `data_berita/`. Kolom yang digunakan adalah `filename`, `title`, dan `content`; judul dan isi artikel ditulis ke file yang sama. Jalankan perintah ini dari direktori proyek.

## Alur pemrosesan

1. Artikel dimuat dari CSV/TSV (skrip terminal) atau dari file teks dalam `data_berita/` (aplikasi web).
2. Teks diproses dengan case folding, pembersihan karakter, stopword removal, dan stemming.
3. Sistem membentuk representasi TF-IDF dan model BM25 dari korpus.
4. Query diproses dengan langkah preprocessing yang sama, lalu dokumen diberi skor dan diurutkan.
5. Untuk setiap hasil, sistem menghitung ringkasan ekstraktif dari isi artikel.
6. Precision@5 dan MRR dihitung menggunakan kategori mayoritas pada daftar hasil sebagai definisi relevansi.

## Catatan interpretasi hasil

- Metrik evaluasi pada implementasi ini **bukan evaluasi relevansi terhadap label query yang diberi anotasi**. Sistem menganggap kategori yang paling sering muncul di antara lima hasil sebagai kategori relevan. Karena itu, Precision@5 dan MRR lebih tepat dipandang sebagai indikator konsistensi kategori hasil, bukan ukuran kualitas pencarian yang tervalidasi.
- Pada `app.py`, kategori diambil dari bagian nama file sebelum karakter `_`. Namun, `split_bbc_csv.py` saat ini menggunakan nama file dari kolom dataset (contohnya `001.txt`) dan tidak menyertakan kolom kategori. Akibatnya, file yang dibuat dengan splitter tersebut tidak membawa label kategori yang sesuai untuk evaluasi aplikasi. Hasil pencarian tetap ditampilkan, tetapi metrik kategorinya tidak dapat diandalkan kecuali nama file dan pemuatan kategori disesuaikan.
- Pada `ir_tfidf.py`, inverted index digunakan untuk memilih kandidat berdasarkan kecocokan token query. Jika tidak ada token yang cocok, seluruh dokumen menjadi kandidat.
- Ringkasan menggunakan ranking kalimat berbasis TF-IDF, bukan model peringkasan generatif.
- Dataset dan preprocessing proyek ini berorientasi pada teks bahasa Inggris.

