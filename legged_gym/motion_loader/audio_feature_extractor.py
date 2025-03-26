import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa.feature
'''
节拍（Beat）:librosa.beat.beat_track()等函数可以提取音乐中的节拍
            机器人的动作在这些节拍上触发，例如跳跃或转动
            持续时间可以设置为下一个节拍与当前节拍的时间差
能量（Energy）：音乐的能量可以反映出某些部分的强烈程度，通常这些部分也是重音或高潮部分
            通过计算音频的均方根能量（RMS）来获得能量特征
            rms = librosa.feature.rms(y=audio)
            根据能量值的大小调整机器人的动作幅度
重音（Onset Strength）:librosa.onset.onset_strength()可以帮助提取音频中的重音信息
                    表示节奏强度（Onsets）在这些重音位置上触发机器人的特殊动作
频谱特征（Spectral Features）：频谱特征（如频谱质心、频谱带宽等）表示了音乐中频率的分布情况，反映音乐的音色
                            librosa.feature.spectral_centroid()等函数可以提取频谱特征
MFCC（梅尔频率倒谱系数）：反映了音乐的音色特征
                    librosa.feature.mfcc()可以提取音乐的MFCC特征
# 使用示例：
audio_extractor = AudioFeatureExtractor()
'''


class AudioFeatureExtractor:
    def __init__(self):
        # 加载音频文件
        audio_path = '/home/lw/音乐/beat8swav.wav'
        self.audio_path = audio_path
        self.audio, self.sr = librosa.load(audio_path)
        self.duration = librosa.get_duration(y=self.audio, sr=self.sr)
        print(f"Audio duration: {self.duration} seconds")

    def get_tempo_and_beats(self, max_episode_length_s):
        # 提取节拍信息
        tempo, beat_frames = librosa.beat.beat_track(y=self.audio, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        beat_times = beat_times[beat_times <= max_episode_length_s+1]
        # beat_times_before = beat_times.copy()
        # # 将所有节拍时间前移 0.1 秒
        # beat_times -= 0.1

        print(f'Tempo: {tempo} BPM')
        # print(f'Number of beats detected: {len(beat_frames)}')
        print(f'Beat times: {beat_times}')
        return tempo, beat_times

    def get_pitch(self, max_episode_length_s):
        # 限制音频长度
        y = self.audio[:int(max_episode_length_s * self.sr)]  # 限制音频信号的长度

        # 提取音高特征（使用librosa的pyin方法）
        pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('B7'))

        # 将音高数据转为时间
        pitch_times = librosa.times_like(pitch)

        # print(f'Pitch times: {pitch_times}')
        print(f'Pitch values: {pitch}')

        return pitch_times, pitch

    def plot_beat_tracking(self):
        # 绘制音频波形图并标注节拍
        tempo, beat_times = self.get_tempo_and_beats()
        plt.figure(figsize=(10, 6))
        librosa.display.waveshow(self.audio, sr=self.sr, alpha=0.5)
        plt.vlines(beat_times, -1, 1, color='r', alpha=0.7, label='Beats')
        plt.title('Beat Tracking')
        plt.legend()
        plt.show()

    def get_rms_energy(self, max_episode_length_s):
        # 提取均方根能量
        rms = librosa.feature.rms(y=self.audio)[0]
        # 将时间戳转换为对应的音频帧索引
        frame_times = librosa.frames_to_time(range(len(rms)), sr=self.sr)

        # 筛选出在 max_episode_length_s 范围内的音频帧
        rms = rms[frame_times <= max_episode_length_s]
        frame_times = frame_times[frame_times <= max_episode_length_s]

        # 打印音量信息
        print(f"RMS Energy: {rms}")
        print(f"Frame times corresponding to RMS: {frame_times}")

        return rms, frame_times

    def plot_rms_energy(self):
        # 绘制 RMS 能量图
        rms = self.get_rms_energy()
        time = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr)
        plt.figure(figsize=(10, 6))
        plt.plot(time, rms, label='RMS Energy')
        plt.title('RMS Energy Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS')
        plt.legend()
        plt.show()

    def plot_spectrogram(self):
        # 计算并绘制频谱图
        S = np.abs(librosa.stft(self.audio))
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.show()

    def get_mfcc(self, n_mfcc=13):
        # 提取并返回 MFCC 特征
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=n_mfcc)
        print(f"MFCCs: {mfccs}")
        return mfccs

    def plot_mfcc(self, n_mfcc=13):
        # 可视化 MFCC 特征
        mfccs = self.get_mfcc(n_mfcc)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.xlabel('Time (s)')
        plt.ylabel('MFCC Coefficients')
        plt.show()
