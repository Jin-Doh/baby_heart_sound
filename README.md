## PyTorch를 활용한 태아 심장 소리 유사성 분석

이 프로젝트는 추석에 가족들끼리 이야기하다 장난으로 시작되었습니다.
PyTorch를 사용하여 태아 심장 소리와 기차 소리, 말발굽 소리의 유사성을 비교하는 간단한 예제입니다. MFCC(Mel-Frequency Cepstral Coefficients) 특징 추출 및 코사인 유사도 측정을 통해 어떤 소리와 더 유사한지 판별합니다.

### 주요 기능

* **오디오 파일 로드:** WAV 및 M4A 형식의 오디오 파일을 로드합니다. M4A 파일은 WAV로 변환하여 처리합니다.
* **소리 정규화:** 입력 소리의 길이를 5초로 통일하고, 최댓값을 1, 최솟값을 0으로 정규화하여 일관된 분석을 가능하게 합니다.
* **MFCC 특징 추출:** `torchaudio` 라이브러리를 사용하여 MFCC 특징 벡터를 추출합니다.
* **유사도 비교:** MFCC 특징 벡터를 정규화하고 길이를 맞춘 후, 코사인 유사도를 계산하여 두 소리의 유사성을 측정합니다.
* **결과 출력:** 태아 심장 소리가 기차 소리와 말발굽 소리 중 어느 소리와 더 유사한지, 그리고 얼마나 더 유사한지 백분율로 출력합니다.

### 실행 방법

1. 필요한 라이브러리를 설치합니다:
   ```bash
   pip install pydub torchaudio
   ```

2. `resource` 폴더에 다음 파일들을 준비합니다:
   * `audio.wav`: 태아 심장 소리 파일
   * `horse.mp3`: 말발굽 소리 파일
   * `train.mp3`: 기차 소리 파일

3. 코드를 실행합니다:
   ```bash
   python main.py
   ```

### 참고 사항

* 이 코드는 간단한 예제이며, 실제 태아 심장 소리 분석에는 더 정교한 전처리 및 분석 기법이 필요할 수 있습니다.
* DTW(Dynamic Time Warping)와 같은 고급 유사도 측정 방법을 사용하여 더 정확한 결과를 얻을 수도 있습니다.
* 의료 분야와 관련된 중요한 작업이므로, 전문 지식과 함께 신중하게 접근해야 합니다.

### 개선 가능성

* 다양한 특징 추출 방법(예: mel-spectrogram, chroma features)을 시도하여 분석 성능을 향상시킬 수 있습니다.
* 더 많은 데이터를 사용하여 모델을 학습시키고, 딥러닝 기반의 접근 방식을 적용할 수도 있습니다.
* 실제 환경에서의 노이즈 처리 및 다양한 상황에 대한 모델의 견고성을 높이는 연구가 필요합니다.