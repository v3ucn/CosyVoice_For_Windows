import torchaudio

# 加载音频文件
waveform, sample_rate = torchaudio.load("Keira.wav")

print(waveform)

# 定义效果链
effects = [
    ["tempo", "1.2"],  # 将速度提高50%
    ["rate", f"{sample_rate}"]  # 保持原始采样率
]

# 应用效果
augmented_waveform, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
    waveform, 
    sample_rate, 
    effects
)


# 保存为新的WAV文件
torchaudio.save("output.wav", augmented_waveform, new_sample_rate)