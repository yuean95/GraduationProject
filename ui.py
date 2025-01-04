import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
# 匯入資料預處理程式碼
from audio_processing import audio_processing

# 加載訓練好的 CNN 模型
model = load_model('bird_sound_cnn_model.h5')  # 模型路徑
labels = ["原鴿", "麻雀", "綠繡眼"]  # 分類標籤，順序跟模型的標籤編碼對應

# 輸出梅爾頻譜圖的路徑
mel_path = "mel_spectrogram.png"

class BirdSoundClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("鳥鳴聲分類器")
        
        # 添加按鈕讓用戶選擇音檔
        self.choose_button = tk.Button(root, text="選擇音檔", command=self.load_audio_file)
        self.choose_button.pack(pady=10)
        
        # 添加用於顯示梅爾頻譜圖的空白
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.figure, root)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # 用於顯示結果的部分，此處為ui打開時的預設
        self.result_label = tk.Label(root, text="請選擇一個音檔進行分析", font=("Arial", 12))
        self.result_label.pack(pady=10)

    def load_audio_file(self):
        # 打開文件對話框選擇音檔
        file_path = filedialog.askopenfilename(filetypes=[("音頻文件", "*.wav *.mp3")])
        if file_path:
            # 處理音檔並生成梅爾頻譜圖
            audio_processing(audio_path=file_path, output_mel_path=mel_path, target_length=15)
            
            # 顯示梅爾頻譜圖
            self.display_mel_spectrogram(mel_path)
            
            # 使用模型進行分類
            self.classify_sound(mel_path)

    def display_mel_spectrogram(self, mel_path):
        # 清空之前的圖
        self.ax.clear()
        # 加載並顯示梅爾頻譜圖
        mel_img = plt.imread(mel_path)
        self.ax.imshow(mel_img)
        self.ax.axis('off')
        self.canvas.draw()

    def classify_sound(self, mel_path):
        # 加載梅爾頻譜圖
        mel_img = plt.imread(mel_path)
        mel_img_resized = np.resize(mel_img, (1, 558, 326, 3))  # 將載入的圖轉成模型輸入格式
        predictions = model.predict(mel_img_resized)    # 得到每個標籤的概率
        predicted_label = labels[np.argmax(predictions)]    # 找到概率最高的標籤
        probability = np.max(predictions) * 100 # 將最高概率乘以100，用於顯示百分比
        # 顯示結果
        self.result_label.config(text=f"模型預測：{predicted_label} (概率：{probability:.2f}%)")

# 運行應用程式
if __name__ == "__main__":
    root = tk.Tk()  # 創建tkinter的視窗
    UI = BirdSoundClassifierUI(root)   # 啟動UI
    root.mainloop() # 保持UI視窗開啟
