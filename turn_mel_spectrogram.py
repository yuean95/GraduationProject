import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def turn_mel_spectrogram(audio_path, output_mel_path, plot=False):
    """
    頻譜圖繪製函數。
    直接套librosa自帶的梅爾倒頻譜函式。
    
    參數:
    - audio_path: 輸入音訊檔案的路徑 (這裡理論上應該要是去噪過後的 WAV檔)
    - output_mel_path: 儲存梅爾頻譜圖圖片的路徑 (png 格式)
    - plot: 是否顯示頻譜圖繪製結果，用於除錯
    """
    # 讀取音檔
    y, sr = librosa.load(audio_path)
    
    # 計算梅爾頻譜，n_mel是梅爾濾波器組的數量，hop_length是相鄰幀之間的時間間隔，n_fft是窗口的長度，默認是2048
    # hop_length跟n_fft共同決定了幀跟幀之間的重複程度
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    # 將功率譜轉成dB(轉dB會有更直觀的視覺效果)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 繪製梅爾頻譜圖
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, hop_length=512, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(output_mel_path)

    # 是否顯示畫好的圖
    if plot:
        plt.show()
    else:
        plt.close()

# 函式測試
turn_mel_spectrogram("output.wav", "mel_spectrogram.png", plot=True)
