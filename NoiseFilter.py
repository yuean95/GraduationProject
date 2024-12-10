import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def noise_reduction(audio_path, output_path, noise_reduction_factor=0.5, plot=False):
    """
    雜音過濾函數。
    原理:
        1.原始音頻 轉成 STFT：把音頻分解成時間和頻率的矩陣。
        2.幅值與相位分解：從複數提取幅值和相位。
        3.門檻去噪：設個門檻，低於門檻的頻率就去掉。
        4.反STFT：轉回時域。
    
    
    參數:
    - audio_path: 輸入音訊檔案的路徑 (支持 MP3, WAV等)
    - output_path: 儲存處理後音訊的路徑 (WAV格式)
    - noise_reduction_factor: 噪音抑制強度 (0 到 1，數值越大過濾越強)
    - plot: 是否繪製頻譜圖，用於除錯
    """
    # 讀取音訊檔案，y:音頻時間序列，sr:音頻採樣率(默認22050，None則採用原檔採樣率)
    y, sr = librosa.load(audio_path)
    
    # 計算音訊的短時傅立葉變換 (STFT)
    stft = librosa.stft(y)
    # 計算頻率的幅值跟相位
    magnitude, phase = np.abs(stft), np.angle(stft)
    
    # 根據幅值計算噪音門檻
    noise_threshold = np.mean(magnitude) * noise_reduction_factor
    # 低於門檻的頻率設為0 (相當於去除噪音)
    magnitude_denoised = np.where(magnitude > noise_threshold, magnitude, 0)
    
    # 恢復時域信號
    stft_denoised = magnitude_denoised * np.exp(1j * phase)
    y_denoised = librosa.istft(stft_denoised)
    
    # 將處理後的音訊儲存為.wav
    sf.write(output_path, y_denoised, sr)
    
    #繪製頻譜圖，用於除錯，正式使用時把plot這參數設為false，或乾脆刪掉也行
    if plot:
        # 繪製頻譜對比
        plt.figure(figsize=(12, 6))
        
        # 原始音訊頻譜
        plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                                 sr=sr, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('before')
        
        # 處理後音訊頻譜
        plt.subplot(2, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(magnitude_denoised, ref=np.max),
                                 sr=sr, y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('after')
        
        plt.tight_layout()
        plt.show()

# 函數測試
noise_reduction("test.mp3", "output.wav", noise_reduction_factor=0.5, plot=True)
