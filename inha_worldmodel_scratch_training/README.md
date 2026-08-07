# INHA AI Challenge

이 패키지는 대회 참가용 데이터, 제출 파일 생성 도구, 베이스라인 예시를 포함합니다.

## 폴더 구성

- `data/`: 학습 데이터와 평가 입력 데이터
- `submission_kit/`: 생성한 영상을 제출 CSV로 변환하는 도구
- `baseline/`: 베이스라인 모델 실행 예시
- `sample_submission.csv`: 제출 CSV 형식 예시

## 제출 파일 만들기

참가자는 `data/eval/images`와 `data/eval/actions`를 입력으로 사용해 평가 샘플별 영상을 생성합니다.

Python 3.8-3.12 환경을 지원합니다.

생성한 mp4 파일은 아래 폴더에 넣어 주세요.

```text
submission_kit/input_videos/
```

파일명은 평가 샘플 ID와 같아야 합니다.

```text
submission_kit/input_videos/sample_000000.mp4
submission_kit/input_videos/sample_000001.mp4
...
submission_kit/input_videos/sample_000215.mp4
```

그 다음 아래 명령을 실행합니다.

```bash
cd submission_kit
python -m pip install -r requirements.txt
python make_submission_csv.py
```

제출 파일은 아래 경로에 생성됩니다.

```text
submission_kit/submission_features.csv
```

이 CSV 파일을 데이콘에 제출하면 됩니다.

## CSV 형식

제출 CSV 컬럼은 아래와 같습니다.

```text
sample_id,feature_component,feature_json
```

`sample_submission.csv`에서 제출 형식 예시를 확인할 수 있습니다.

## 베이스라인 사용

베이스라인을 실행해 보고 싶은 경우 `baseline/baseline.ipynb`를 사용하세요.

베이스라인은 선택 사항입니다. 자체 모델을 사용하는 경우에는 `baseline/` 폴더를 사용하지 않아도 됩니다.
