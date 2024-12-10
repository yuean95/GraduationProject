import librosa
import librosa.display
import matplotlib.pyplot as plt

def turn_MFCC(audio_path, output_path, plot=False):
    """
    頻譜圖繪製函數。
    直接套librosa自帶的MFCC。
    
    參數:
    - audio_path: 輸入音訊檔案的路徑 (這裡理論上應該要是去噪過後的 WAV檔)
    - output_path: 儲存圖片的路徑 (png 格式)
    - plot: 是否顯示頻譜圖繪製結果，用於除錯

    """
    # 讀取音檔
    file_path = 'output.wav'
    # 讀取音訊檔案，y:音頻時間序列，sr:音頻採樣率(默認22050，None則採用原檔採樣率)
    y, sr = librosa.load(file_path)
    
    # 計算MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 繪製圖形
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, hop_length=512, cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title('MFCC Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficients')
    plt.tight_layout()

    # 儲存圖片
    plt.savefig(output_path)

    # 是否顯示畫好的圖
    if plot:
        plt.show()
    else:
        plt.close()

# 函式測試
turn_MFCC("output.wav", "mfcc.png", plot=True)
