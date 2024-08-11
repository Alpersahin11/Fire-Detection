import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd


egitim_dizin = 'yangın/Train_Data/'
dogrulama_dizin = 'yangın/Val_Data/'
test_dizin = 'yangın/Test_Data/'


egitim_generator = ImageDataGenerator(rescale=1.0 / 255)
dogrulama_generator = ImageDataGenerator(rescale=1.0 / 255)

# Eğitim ve Doğrulama
egitim_jenerator = egitim_generator.flow_from_directory(
    egitim_dizin,
    target_size=(150, 150),
    batch_size=64,  # Önerilen batch size
    class_mode='binary'
)

dogrulama_jenerator = dogrulama_generator.flow_from_directory(
    dogrulama_dizin,
    target_size=(150, 150),
    batch_size=64,  # Önerilen batch size
    class_mode='binary'
)

# Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),  # Ekstra konvolüsyon katmanı
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout ekleyerek overfitting'i önleyin
    layers.Dense(1, activation='sigmoid')  # Binary classification için sigmoid
])


optimizasyon = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizasyon,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# EarlyStopping
durdurma = EarlyStopping(
    monitor='val_loss',  # Doğrulama kaybını izleyin
    patience=20,  # Doğrulama kaybı 20 epoch boyunca iyileşmezse eğitimi durdurun
    restore_best_weights=True  # En iyi epoch'taki ağırlıkları geri yükleyin
)

# Model eğitimi
gecmis = model.fit(
    egitim_jenerator,
    steps_per_epoch=egitim_jenerator.samples // egitim_jenerator.batch_size,
    epochs=100,
    validation_data=dogrulama_jenerator,
    validation_steps=dogrulama_jenerator.samples // dogrulama_jenerator.batch_size,
    callbacks=[durdurma]
)

# Eğitim ve doğrulama sonuçları
gecmis_data = pd.DataFrame(gecmis.history)
print(gecmis_data)


degerlendirme_sonuclari = model.evaluate(dogrulama_jenerator)
print(f'Test kaybı: {degerlendirme_sonuclari[0]}')
print(f'Test doğruluğu: {degerlendirme_sonuclari[1]}')


model.save('yangin_tespit_modeli.h5')  # TensorFlow formatında kaydedin
print("Model başarıyla .h5 formatında kaydedildi.")

# Eğitim ve doğrulama kaybı/doğruluk grafikleri<
plt.figure(figsize=(12, 4))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(gecmis_data['accuracy'], label='Eğitim Doğruluk')
plt.plot(gecmis_data['val_accuracy'], label='Doğrulama Doğruluk')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(gecmis_data['loss'], label='Eğitim Kayıp')
plt.plot(gecmis_data['val_loss'], label='Doğrulama Kayıp')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.show()
