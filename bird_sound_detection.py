import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 讀取梅爾頻譜圖資料
DATA_DIR = 'C:/Users/USER/Desktop/畢業專題/mel spectrogram'  # 替換為梅爾頻譜圖資料夾路徑
IMAGE_SIZE = (558, 326)  # 定義輸入圖片大小(=預處理出來的圖的尺寸)

# 讀取圖片與標籤
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):  # 每個子資料夾表示一個類別
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if file_path.endswith('.png') or file_path.endswith('.jpg'):
                    img = plt.imread(file_path)
                    img = np.resize(img, (*IMAGE_SIZE, 3))  # 將img調整到固定的大小(558, 326)並保留3個顏色通道(RGB)
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

# 加載資料
images, labels = load_data(DATA_DIR)

# 將標籤編碼轉為數字
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)  # One-hot 編碼

# 分割資料集，訓練集:測試集=:8:2
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 架構 CNN 模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),  # 第一層卷積，採ReLU激活函數
    MaxPooling2D((2, 2)),  # 最大池化層

    Conv2D(64, (3, 3), activation='relu'),  # 第二層卷積
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),  # 第三層卷積
    MaxPooling2D((2, 2)),

    Flatten(),  # 展平層
    Dense(128, activation='relu'),  # 全連接層，128個神經元，ReLU
    Dropout(0.5),  # 防止過擬合
    Dense(labels.shape[1], activation='softmax')  # 輸出層，根據類別數進行分類，採softmax激活函數
])

# 編譯模型，優化器用adam，損失函數用
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練模型，10個訓練週期
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 儲存模型
model.save('bird_sound_cnn_model.h5')

# 評估模型，損失值越小越好，準確率越高越好
loss, accuracy = model.evaluate(X_test, y_test)
print(f"測試集損失值: {loss}, 測試集準確率: {accuracy}")

# 繪製損失值
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='test loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()

# 繪製準確率
plt.figure()
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='test accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()
