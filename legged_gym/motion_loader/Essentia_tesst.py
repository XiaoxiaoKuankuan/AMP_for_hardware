import essentia.standard as es

audio_path = '/home/lw/音乐/beat8swav.wav'

# 1. 加载音频文件（替换为你的文件路径）
audio = es.MonoLoader(filename=audio_path, sampleRate=44100)()

# 2. 提取节奏特征
rhythm_extractor = es.RhythmExtractor2013()
bpm, beats, confidence, _ , _ = rhythm_extractor(audio)
print(f"音乐BPM：{bpm} | 节奏置信度：{confidence}")
print(f"节拍时间点(秒): {beats}")
