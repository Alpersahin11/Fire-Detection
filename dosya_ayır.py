import os
import shutil
import random

# Kaynak ve hedef dizinler
source_dir = 'yangın/All_Data/Non_Fire'
train_dir = 'yangın/Train_Data/Non_Fire'
val_dir = 'yangın/Val_Data/Non_Fire'
test_dir = 'yangın/Test_Data/Non_Fire'

# Hedef dizinleri oluştur
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Dosyaların listesini al
files = os.listdir(source_dir)

# Dosyaları karıştır
random.shuffle(files)

# Veriyi böl
total_files = len(files)
train_size = int(total_files * 0.75)
val_size = int(total_files * 0.20)
test_size = total_files - train_size - val_size

# Eğitim, doğrulama ve test dosyalarını ayır
train_files = files[:train_size]
val_files = files[train_size:train_size + val_size]
test_files = files[train_size + val_size:]

# Dosyaları hedef dizinlere kopyala
for file in train_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))

for file in test_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))

print("Dosyalar başarıyla dağıtıldı.")
