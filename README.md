# Lyra 데이터셋 도구

[Experiment 1 실행 가이드](docs/experiment1.md)에서 VLSBench 장르 실험의 WSL 계획 확인, GPU 추론, 로컬 판정, CSV 분석 절차를 볼 수 있습니다.

[DGX Spark StyleVector · EmoStyle 실행 가이드](docs/continuous-style.md)에는 두 Conda 환경의 설치, 체크포인트 준비, 단일 샘플 Smoke Test 명령이 있습니다.

Lyra는 VLSBench와 VLSU의 원본 메타데이터 및 이미지를 로컬에 저장하고, Python에서 샘플 단위로 읽습니다. WSL과 Linux 모두 `lyra` Conda 환경에서 실행합니다. 새 환경에는 Python 3.11을 설치하며, 기존 `lyra` 환경을 재사용하려면 Python 3.10 이상이 필요합니다. GPU, CUDA, PyTorch는 필요하지 않습니다.

| 데이터셋 | 원본 메타데이터 | 이미지 | 로더의 라벨 키 |
| --- | --- | --- | --- |
| [VLSBench](https://huggingface.co/datasets/Foreshhh/vlsbench) | `vlsbench/data.json` | `vlsbench/imgs/` 내 원본 경로 | `category`, `sub_category` |
| [VLSU](https://github.com/apple-aiml-research/ml-vlsu) | `vlsu/data/VLSU.csv` | `vlsu/images/<uuid>` | `image_grade`, `consensus_text_grade`, `consensus_combined_grade`, `image_category`, `text_category`, `combined_category` |

VLSBench는 원본 JSON과 해당 JSON이 참조하는 개별 이미지를 내려받습니다. 배포 저장소의 `imgs.tar` 및 Parquet 사본은 생략합니다. VLSU는 CSV의 `web_path`에서 가져온 이미지 바이트를 확장자 없는 UUID 파일에 저장합니다. 이미지 형식은 파일 내용으로 식별할 수 있으며, CSV의 URL과 빈 라벨 값은 그대로 보존됩니다. [VLSBench 파일 목록](https://huggingface.co/datasets/Foreshhh/vlsbench/tree/main) · [VLSU 원본 CSV](https://github.com/apple-aiml-research/ml-vlsu/blob/main/data/VLSU.csv)

## WSL에서 설치, 다운로드, 로딩하기

Conda를 사용할 수 있는 WSL 셸에서 실행합니다. `lyra` 환경이 이미 있으면 생성 명령을 건너뛰고 활성화합니다. `DATA_DIR`에는 원하는 저장 경로를 지정할 수 있습니다. 저장 경로를 바꾼 뒤에도 다운로드와 로더에 같은 값을 전달합니다.

```bash
cd /home/six/workspace/Lyra
conda create -n lyra python=3.11 pip
conda activate lyra
python --version
python -m pip install -r requirements.txt

DATA_DIR=./data
python -m lyra.download vlsbench --data-dir "$DATA_DIR"
python -m lyra.download vlsu --data-dir "$DATA_DIR" --attempts 3

python -m lyra.datasets vlsbench --data-dir "$DATA_DIR"
python -m lyra.datasets vlsu --data-dir "$DATA_DIR"
```

로딩 명령은 메타데이터 행과 로컬 이미지의 수를 표시하고, 이미지가 있는 첫 샘플의 ID, 경로, 라벨을 출력합니다. 로컬 이미지가 한 장도 없으면 종료 코드 1을 반환합니다.

## 중단 복구와 리비전 고정

- 중단되거나 실패하면 같은 명령을 실행합니다. 저장된 파일은 건너뛰고 누락된 파일을 내려받습니다. VLSU는 응답을 임시 파일에 받은 뒤 이미지 형식을 식별하고, 다운로드한 바이트를 최종 경로에 저장합니다. 연결·전송 오류, HTTP 408·429·5xx는 `--attempts` 범위 내에서 재시도합니다. `--workers`로 동시 이미지 요청 수를 지정할 수 있습니다.
- 이미지를 받지 못한 행은 데이터셋별 `download_failures.jsonl`에 실행 시각, 샘플 ID, 원본 URL 또는 경로, 오류와 함께 누적됩니다. 이번 실행에 실패가 있으면 다운로드 명령은 종료 코드 1을 반환합니다. VLSU는 외부 호스트의 URL이 만료되거나 접근을 거부할 수 있습니다. 원본 CSV가 저장된 이후에는 사용 가능한 이미지만 연결한 상태로 전체 메타데이터를 읽을 수 있습니다.
- 데이터셋별 `source.json`에 원본 저장소 URL, 요청한 리비전, 해석된 전체 커밋 SHA를 기록합니다. 첫 실행에서 `--revision`을 생략하면 `main`을 사용합니다. 이후에는 기록된 커밋을 사용합니다. 다른 리비전을 받으려면 새 `--data-dir`을 지정합니다.
- 서버에서 동일한 메타데이터 버전을 받으려면 각 `source.json`의 `commit` 값을 해당 다운로드 명령의 `--revision`에 전달합니다. VLSU의 커밋으로 고정되는 대상은 CSV입니다. 외부 이미지 URL의 응답은 호스트의 상태에 따라 달라질 수 있습니다.

## Python에서 읽기

저장소 루트에서 Python을 실행합니다. 각 함수는 원본 순서대로 인덱싱할 수 있는 `list[Sample]`을 반환합니다. 이미지가 아직 없으면 `image_path`가 `None`입니다. `original_metadata`에는 원본 JSON 레코드 또는 CSV 행의 모든 필드가 들어 있습니다.

```python
from lyra.datasets import load_vlsbench, load_vlsu

bench_samples = load_vlsbench("./data")
vlsu_samples = load_vlsu("./data")

sample = vlsu_samples[0]
sample.sample_id
sample.image_path
sample.instruction
sample.safety_labels
sample.original_metadata
```

이미지를 열어야 할 때는 `image_path`가 있는 샘플에 대해 `Pillow`의 `Image.open(sample.image_path)`를 사용할 수 있습니다.

## WSL에서 GitHub에 올리기

먼저 로컬 저장소 상태를 확인합니다. 기존 저장소면 현재 브랜치와 원격 주소를 확인합니다.

```bash
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse --show-toplevel
  git status --short --branch
  git remote -v
else
  ls -la
fi
```

처음 확인한 Lyra 디렉터리에는 Git 저장소가 없었습니다. 위 검사에서 여전히 저장소가 없을 때만 초기화합니다. 기존 저장소가 표시되면 최상위 경로가 의도한 Lyra 저장소인지 확인합니다.

```bash
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi

TARGET=https://github.com/p6trick/tmp_lyra.git
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$TARGET"
fi
git remote -v
```

`origin`이 다른 저장소를 가리키면 기존 설정을 유지하고, 사용 중이지 않은 별도 이름으로 이 작업의 원격을 추가합니다. 예를 들어 `git remote add lyra-publish "$TARGET"`로 추가한 경우 아래 `PUBLISH_REMOTE`를 `lyra-publish`로 지정합니다.

```bash
git add -- .gitignore README.md requirements.txt lyra/__init__.py lyra/datasets.py lyra/download.py
git status --short --branch
git diff --cached --stat
```

스테이징 내용을 확인한 뒤 지정한 파일만 커밋합니다. 데이터와 캐시는 로컬에 보관합니다. 기존 이력이 생긴 저장소에서 푸시하는 경우 현재 브랜치의 이전 커밋도 함께 올라가므로 이력을 확인합니다.

Git 작성자 정보가 없다면 아래 예시를 본인의 이름과 GitHub 계정에 등록한 이메일로 바꿔 설정합니다. 이메일을 공개하지 않으려면 GitHub 이메일 설정에 표시되는 본인의 `noreply` 주소를 사용할 수 있습니다. `--local` 설정은 현재 저장소에 적용됩니다.

```bash
git config --local user.name "Your Name"
git config --local user.email "you@example.com"
```

```bash
TARGET=https://github.com/p6trick/tmp_lyra.git
PUBLISH_REMOTE=origin
if git commit --only -m "Add VLSBench and VLSU dataset tools" -- \
  .gitignore README.md requirements.txt lyra/__init__.py lyra/datasets.py lyra/download.py
then
  if [ "$(git remote get-url --push --all "$PUBLISH_REMOTE")" = "$TARGET" ]; then
    git push "$PUBLISH_REMOTE" HEAD:refs/heads/main
  else
    printf '%s\n' 'Push URL을 확인하고 PUBLISH_REMOTE를 지정하세요.'
  fi
fi
```

## Linux 서버에서 재사용하기

Conda를 사용할 수 있는 Linux 셸에서 작업할 상위 디렉터리로 이동한 뒤 실행합니다. `lyra` 환경이 이미 있으면 생성 명령을 건너뛰고 활성화합니다.

```bash
git clone https://github.com/p6trick/tmp_lyra.git Lyra
cd Lyra
conda create -n lyra python=3.11 pip
conda activate lyra
python --version
python -m pip install -r requirements.txt

DATA_DIR=./data
python -m lyra.download vlsbench --data-dir "$DATA_DIR"
python -m lyra.download vlsu --data-dir "$DATA_DIR" --attempts 3

python -m lyra.datasets vlsbench --data-dir "$DATA_DIR"
python -m lyra.datasets vlsu --data-dir "$DATA_DIR"
```

## 원본 라이선스

- VLSBench는 Apache-2.0으로 표기됩니다. [Hugging Face 데이터셋 카드](https://huggingface.co/datasets/Foreshhh/vlsbench)
- VLSU 데이터는 CC BY-NC-ND로 표기됩니다. [Apple 데이터 라이선스](https://github.com/apple-aiml-research/ml-vlsu/blob/main/LICENSE_DATA)
