import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Modeli yükleme
model = load_model('fire_detection_model_son2.h5')

# Test edilecek resmi yükleme
img_path = 'yangın/Test_Data/Fire/F_1571.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Tahmin yap
prediction = model.predict(img_array)

# Sonuçları yorumlama
if prediction[0][0] < 0.5:
    label = "Yangın Tespit Edildi!"
    color = 'red'
else:
    label = "Yangın Tespit Edilmedi"
    color = 'green'

# Görüntüyü açma ve kare çizme
img = plt.imread(img_path)
fig, ax = plt.subplots(1)
ax.imshow(img)

# Eğer yangın tespit edildiyse kare çizme
if prediction[0][0] < 0.5:
    rect = patches.Rectangle((50, 50), 100, 100, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

# Sonuçları görselleştir
plt.title(label)
plt.show()
