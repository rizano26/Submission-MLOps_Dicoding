# Submission 2: Klasifikasi Email Spam dan Non-Spam
Nama: Muhammad Rizano Lukman

Username dicoding: wnefst26

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [E-Mail classification NLP](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification) |
| Masalah | Klasifikasi email menjadi spam dan non-spam adalah tantangan penting dalam manajemen email. Dengan semakin meningkatnya volume email yang diterima setiap hari, penting untuk memilah email yang relevan dan penting dari yang tidak diinginkan. Email spam dapat mengganggu produktivitas, mengancam keamanan dengan phising, dan membebani sistem email. Oleh karena itu, dibutuhkan sistem klasifikasi yang efektif untuk secara otomatis mengidentifikasi email spam sehingga pengguna hanya melihat email yang benar-benar penting di kotak masuk mereka. Sistem ini harus mampu mengenali berbagai pola dan karakteristik email spam, serta terus belajar dan beradaptasi dengan teknik-teknik baru yang digunakan oleh pengirim spam. |
| Solusi machine learning | Membuat Model Deep Learning yang dapat melakukan klasifikasi terhadap E-mail Spam dan Non-Spam. Model ini dibuat dengan target bisa melakukan klasifikasi terhadap mana E-Mail yang spam dan mana E-Mail yang bukan spam. |
| Metode pengolahan | Dataset dilakukan preprocessing data dengan mengubah label dataset dari string menjadi integer (spam == 1 dan non-spam ==0) serta mengubah feature menjadi huruf kecil semua |
| Arsitektur model | Arsitektur model machine learning yang digunakan dalam kode ini adalah model neural network untuk klasifikasi biner, yang dirancang untuk memproses teks dalam fitur Message dan memprediksi kategori (Category). Model ini menggunakan lapisan vektorisasi teks (TextVectorization) untuk mengubah teks mentah menjadi urutan integer, yang kemudian diubah menjadi embedding vektor dengan menggunakan lapisan embedding (Embedding). Vektor embedding ini diratakan menggunakan lapisan GlobalAveragePooling1D untuk mengurangi dimensi dan mempertahankan informasi penting. Setelah itu, lapisan fully connected (Dense) dengan 64 neuron dan aktivasi ReLU diikuti oleh lapisan dengan 32 neuron dan aktivasi ReLU diterapkan untuk menangkap hubungan non-linear dalam data. Output akhir dihasilkan oleh lapisan Dense dengan satu neuron dan aktivasi sigmoid, yang memberikan probabilitas antara 0 dan 1 untuk klasifikasi biner. |
| Metrik evaluasi | Model ini dikompilasi dengan fungsi loss binary_crossentropy dan dioptimalkan menggunakan Adam optimizer dengan tingkat pembelajaran yang ditentukan oleh hyperparameter. Metrik evaluasi yang digunakan adalah BinaryAccuracy, yang mengukur persentase prediksi yang benar dari model dibandingkan dengan label sebenarnya dalam data validasi. |
| Performa model | Model tersebut menghasilkan val_binary_accuracy sebesar 0.9781 dan F1-Score sebesar 0.89 |
| Opsi deployment | Untuk melakukan deployment model, Saya menggunakan service Google Cloud. Saya melakukan push Docker images ke Artifact Registry lalu menjalankannya menggunakan Google Cloud Run. |
| Web app | Berikut adalah link untuk mengakses [spam-detection-model](https://proyek-kedua-ydrkli3j2q-uc.a.run.app/v1/models/spam-detection-model/metadata)|
| Monitoring | Untuk monitoring, saya menggunakan Prometheus yang disinkronkan dengan Grafana. Saya melakukan visualisasi terhadap jumlah request ke model. Dari dashboard Grafana, dapat terlihat bahwa request yang berhasil berjumlah 19 dan request yang invalid berjumlah 3.|
