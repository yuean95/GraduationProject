import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def audio_processing(audio_path, output_mel_path, target_length, noise_reduction_factor=0.5):
    """
    調整音頻長度 -> 噪音過濾 -> 生成梅爾頻譜圖

    參數:
    - audio_path: 輸入音頻檔案的路徑 (MP3)
    - output_mel_path: 儲存梅爾頻譜圖圖片的路徑 (PNG)
    - target_length: 音頻目標長度 (秒)
    - noise_reduction_factor: 噪音抑制強度 (0 到 1，數值越大過濾越強)
    """
    # ===== 音頻長度正規化 =====
    # 讀取音頻
    audio, sr = librosa.load(audio_path)

    # 目標長度（樣本數）
    target_length_samples = int(target_length * sr)

    if len(audio) > target_length_samples:
        # 裁剪音頻
        audio = audio[:target_length_samples]
    else:
        # 填充空白
        pad_length = target_length_samples - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant', constant_values=0)

    # ===== 噪音過濾 =====
    # 計算音訊的短時傅立葉變換 (STFT)
    stft = librosa.stft(audio)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # 根據幅值計算噪音門檻
    noise_threshold = np.mean(magnitude) * noise_reduction_factor
    magnitude_denoised = np.where(magnitude > noise_threshold, magnitude, 0)

    # 恢復時域信號
    stft_denoised = magnitude_denoised * np.exp(1j * phase)
    audio_denoised = librosa.istft(stft_denoised)

    # ===== 梅爾頻譜圖部分 =====
    mel_spec = librosa.feature.melspectrogram(y=audio_denoised, sr=sr, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 繪製並儲存梅爾頻譜圖
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, hop_length=512, cmap='viridis')
    plt.axis('off')
    plt.savefig(output_mel_path, bbox_inches='tight', pad_inches=0)

def batch_processing(input_dir, output_dir, target_length, noise_reduction_factor=0.5):
    """
    批量處理資料夾中的音頻檔案，將每個圖放到令一個資料夾儲存。

    參數:
    - input_dir: 包含音頻檔案的資料夾路徑
    - output_dir: 儲存梅爾頻譜圖圖片的資料夾路徑
    - target_length: 音頻目標長度 (秒)
    - noise_reduction_factor: 噪音抑制強度 (0 到 1，數值越大過濾越強)
    """
    # 檢查輸出資料夾是否存在，如果不存在則創建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍歷資料夾中的所有音頻檔案
    for file_name in os.listdir(input_dir):
        # 檢查是否是mp3
        if file_name.lower().endswith(('.mp3', '.wav')):
            audio_path = os.path.join(input_dir, file_name)
            output_mel_path = os.path.join(output_dir, f"{file_name.split(' ')[0]}.png")    #取xeno的典藏號碼作為圖片名稱
            
            print(f"處理文件: {file_name}")
            audio_processing(audio_path, output_mel_path, target_length, noise_reduction_factor)
            print(f"已儲存梅爾頻譜圖: {output_mel_path}")

# 測試函數
if __name__ == "__main__":
    input_directory = "C:/Users/USER/Desktop/畢業專題/sounds/Passer montanus"  # 音檔的資料夾路徑
    output_directory = "C:/Users/USER/Desktop/畢業專題/mel spectrogram/Passer montanus"  # 儲存梅爾頻譜圖的資料夾路徑
    
    batch_processing(
        input_dir=input_directory,
        output_dir=output_directory,
        target_length=15,
        noise_reduction_factor=0.5
    )
