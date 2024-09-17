import os
import pydub
import torch
import torchaudio

os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "0"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')

def load_audio(audio_path):
    if audio_path.endswith('.m4a'):
        audio = pydub.AudioSegment.from_file(audio_path, format="m4a")
        audio.export(audio_path.replace('.m4a', '.wav'), format="wav")
        audio_path = audio_path.replace('.m4a', '.wav')
    waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
    return waveform, sample_rate

def sound_normalization(waveform):
    # 소리의 길이를 5초로 맞춤
    if waveform.shape[1] < 5 * 16000:
        waveform = torch.nn.functional.pad(waveform, (0, 5 * 16000 - waveform.shape[1]))
    else:
        waveform = waveform[:, :5 * 16000]

    # 소리의 최대값을 1로 정규화
    max_val = waveform.abs().max().item()
    waveform = waveform / max_val

    # 소리의 최소값을 0으로 정규화
    waveform = (waveform + 1) / 2

    return waveform

def extract_features(before_waveform, sample_rate):
    waveform = sound_normalization(before_waveform)
    print(f"원본 소리: {before_waveform.shape}")
    print(f"정규화된 소리: {waveform.shape}")

    mfccs = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={
            "n_fft": 400,
            "hop_length": 160,
            "n_mels": 23,
            "center": False
        }
    )(waveform).to(get_device())
    return mfccs

def compare_similarity(mfccs1, mfccs2):
    # MFCC 데이터 정규화
    mfccs1 = (mfccs1 - mfccs1.mean(dim=2, keepdim=True)) / mfccs1.std(dim=2, keepdim=True)
    mfccs2 = (mfccs2 - mfccs2.mean(dim=2, keepdim=True)) / mfccs2.std(dim=2, keepdim=True)

    # 길이 통일 (짧은 쪽에 맞춰 자르기)
    min_length = min(mfccs1.shape[2], mfccs2.shape[2])
    mfccs1 = mfccs1[:, :, :min_length]
    mfccs2 = mfccs2[:, :, :min_length]

    # 코사인 유사도 계산 (각 채널별로 계산 후 평균)
    similarity = torch.nn.functional.cosine_similarity(mfccs1, mfccs2, dim=2)
    avg_similarity = similarity.mean().item()
    return avg_similarity

if __name__ == "__main__":
    target = "resource/audio.wav"
    horse_sound = "resource/horse.mp3"
    train_sound = "resource/train.mp3"

    target_waveform, target_sample_rate = load_audio(target)
    target_mfccs = extract_features(target_waveform, target_sample_rate)
    print(f"태아 심장 소리 MFCCs: {target_mfccs.shape}")

    horse_waveform, horse_sample_rate = load_audio(horse_sound)
    horse_mfccs = extract_features(horse_waveform, horse_sample_rate)
    print(f"말발굽 소리 MFCCs: {horse_mfccs.shape}")

    train_waveform, train_sample_rate = load_audio(train_sound)
    train_mfccs = extract_features(train_waveform, train_sample_rate)
    print(f"기차 소리 MFCCs: {train_mfccs.shape}\n")

    similarity_horse = compare_similarity(target_mfccs, horse_mfccs)
    print(f"태아 심장 소리와 말발굽 소리의 유사도: {similarity_horse}")
    similarity_train = compare_similarity(target_mfccs, train_mfccs)
    print(f"태아 심장 소리와 기차 소리의 유사도: {similarity_train}\n")

    if similarity_train > similarity_horse:
        percentage = (similarity_train - similarity_horse) / similarity_horse * 100
        percentage = round(percentage, 2)
        print(f"태아 심장 소리는 기차 소리와 {percentage}% 더 유사합니다.")
    else:
        percentage = (similarity_horse - similarity_train) / similarity_train * 100
        percentage = round(percentage, 2)
        print(f"태아 심장 소리는 말발굽 소리와 {percentage}% 더 유사합니다.")
