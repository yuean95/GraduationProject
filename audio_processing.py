import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig(output_mel_path)

# 測試函數
if __name__ == "__main__":
    audio_processing(
        audio_path="test.mp3",
        output_mel_path="mel_spectrogram.png",
        target_length=15,
        noise_reduction_factor=0.5
    )
