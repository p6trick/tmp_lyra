# Experiment 1 · Image-conditioned Artistic Genre Analysis

VLSBench의 2,241개 이미지–질문 쌍과 원본 6개 카테고리를 사용합니다. 아래 세 조건이 모두 동일한 `sample_id`와 이미지 파일을 공유합니다. 전체 계획은 **2,241 × (1 + 13 + 13) = 60,507건**입니다.

| 조건 | Target VLM 입력 | 케이스 수 |
| --- | --- | --- |
| `original` | 원본 이미지와 수정하지 않은 원본 Instruction | 샘플당 1건 |
| `lyra_style` | 원본 이미지와 이미지의 시각적 내용 및 분위기에서 영감을 얻는 장르별 창작 프롬프트 | 샘플당 13건 |
| `instruction_aware` | 원본 이미지와 함께 원본 질문을 해석하고 그 상황과 주제를 반영하는 장르별 창작 프롬프트 | 샘플당 13건 |

장르별 형식은 [genres.json](../lyra/experiment1/genres.json)에, 공통 조건 문구는 [conditions.py](../lyra/experiment1/conditions.py)에 있습니다. 두 창작 조건은 이미지에 대한 사실적 진술을 시각적으로 관찰 가능한 내용에 근거하도록 하며, 문맥상 창작임이 드러나는 허구적 구성·비유·대사를 허용합니다. [장르 확장 참고 논문](https://arxiv.org/abs/2608.13947)

## WSL에서 실행 계획 확인

Lyra 저장소 루트에서 실행합니다. 이 명령은 로컬 메타데이터와 이미지 경로를 읽어 샘플 수, 카테고리, 누락 이미지 수, 예상 케이스 수를 출력합니다. Python 표준 라이브러리만 사용합니다.

```bash
conda activate lyra
python -m lyra.experiment1.infer --data-dir ./data --plan
python -m lyra.experiment1.infer --data-dir ./data --preview-sample-id 0
python -m lyra.experiment1.judge --help
python -m lyra.experiment1.analysis --help
```

전체 실행 전에 `sample_count`가 2241, `case_count`가 60507인지 확인합니다. 누락 이미지가 있으면 대응 케이스는 `image_load` 오류로 기록되며 복구 후 재실행할 수 있습니다. `--limit 2`는 앞의 두 샘플로 만든 54건 계획을 점검할 때 사용합니다.

`--preview-sample-id`는 지정한 샘플의 26개 창작 조건에 대해 추론과 동일한 `target_prompt`를 JSON으로 출력합니다. 해당 텍스트에는 추론 시 모델의 채팅 템플릿이 적용됩니다. 미리보기는 모델과 이미지를 열지 않습니다.

## 서버 환경 준비

두 환경 모두 저장소 루트에서 시작합니다. 아래 명령을 실행할 환경에서 동일한 VLSBench 디렉터리와 쓰기 가능한 출력 경로를 지정합니다.

### DGX Spark · ARM64

NVIDIA Container Toolkit을 사용할 수 있다면 DGX Spark용 PyTorch NGC 컨테이너의 PyTorch/CUDA를 사용합니다. 아래 `26.03-py3` 태그는 NVIDIA의 Spark 멀티모달 가이드에 있는 예시입니다. 호스트 드라이버와 사용할 태그의 호환성을 확인하세요. 컨테이너 안에서 PyTorch를 교체하지 않습니다. [DGX Spark 가이드](https://docs.nvidia.com/dgx/dgx-spark/) · [NVIDIA 멀티모달 가이드](https://build.nvidia.com/playbooks/multi-modal-inference/overview) · [NGC 이미지 릴리스 노트](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-03.html)

```bash
uname -m
nvidia-smi
docker run --rm -it --gpus device=0 --ipc=host \
  -v "$PWD:/work" -w /work -e HF_HOME=/work/.cache/huggingface \
  nvcr.io/nvidia/pytorch:26.03-py3 bash
```

이 명령은 현재 저장소를 컨테이너에 마운트합니다. 데이터가 저장소 밖에 있으면 해당 디렉터리도 마운트하고 컨테이너에서 보이는 경로를 `DATA_DIR`로 사용하세요. 호스트에서 출력 파일을 편집할 때는 컨테이너의 사용자 권한을 확인하세요.

```bash
python -m pip install -r requirements-experiment1.txt
python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

### RTX 3090 24GB · x86_64

Conda를 사용할 수 있는 서버에서 환경을 준비합니다. 아래는 CUDA 12.8 휠 예시입니다. 드라이버 환경이 다르면 공식 선택표에서 사용할 CUDA 빌드와 짝이 맞는 PyTorch·torchvision을 설치합니다. [PyTorch 버전별 설치 명령](https://pytorch.org/get-started/previous-versions/) · [NVIDIA CUDA 드라이버 호환성](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

```bash
uname -m
nvidia-smi
conda create -n lyra python=3.11 pip
conda activate lyra
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-experiment1.txt
CUDA_VISIBLE_DEVICES=0 python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

기존 `lyra` 환경이 있으면 환경 생성 명령을 생략합니다. 첫 실행은 batch size 1에서 시작하세요. 기본 이미지 픽셀 예산은 524,288이고 최대 생성 길이는 256토큰입니다. 메모리가 부족하면 `--max-pixels 262144`, `--max-new-tokens 128` 같이 낮출 수 있습니다. BF16을 사용할 수 없는 경우 `--dtype fp16`을 선택할 수 있습니다.

## Target VLM 추론

DGX Spark 컨테이너와 RTX 3090 서버에 공통으로 사용하는 명령입니다. `CUDA_VISIBLE_DEVICES`에는 해당 셸에서 사용할 GPU 한 장을 지정합니다. 대상 모델은 기본값이 Qwen3-VL-8B-Instruct이며, 로컬 Qwen3-VL 체크포인트나 다른 Hub 리비전은 `--model`, `--revision`으로 지정합니다. 저장된 원본 파일은 그대로 읽고, 메모리에서 RGB로 디코딩한 뒤 모든 조건에 동일한 이미지 프로세서를 적용합니다. 기본 디코딩은 BF16, greedy, SDPA입니다. `--attention eager`도 지원합니다. [Qwen 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

```bash
DATA_DIR=./data
RUN_DIR=./outputs/experiment1/full
JUDGE_DIR=./outputs/experiment1/shieldstral
REPORT_DIR=./outputs/experiment1/analysis
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"

# 작은 파일럿
CUDA_VISIBLE_DEVICES=0 python -m lyra.experiment1.infer \
  --data-dir "$DATA_DIR" --output-dir ./outputs/experiment1/pilot --limit 2

# 전체 추론
CUDA_VISIBLE_DEVICES=0 python -m lyra.experiment1.infer \
  --data-dir "$DATA_DIR" --output-dir "$RUN_DIR" \
  --batch-size 1 --max-pixels 524288 --max-new-tokens 256 --dtype bf16 --seed 42
```

`run.json`은 데이터 메타데이터 SHA-256, 데이터셋 출처, 장르 및 프롬프트 해시, 계획된 샘플, 모델 커밋, 실행 설정을 기록합니다. 로컬 모델 경로를 선택했다면 해당 체크포인트를 실행 기간 동안 고정해 두세요. `inference.jsonl`은 `case_id`, `sample_id`, 원본 메타데이터, 원본 질문, 생성에 사용한 `target_prompt`, 모델 템플릿이 렌더링한 `chat_prompt`, 응답, 실제 디코딩 설정 및 오류를 기록합니다. 이미지 경로는 `DATA_DIR/vlsbench` 기준입니다.

같은 설정과 출력 디렉터리에서 재실행하면 성공 케이스를 건너뛰고 나머지를 시도합니다. 재개 전에 `run.json`의 프롬프트 해시와 설정을 검증합니다. 공통 문구나 장르 템플릿을 바꾼 뒤에는 새 출력 디렉터리를 지정합니다. 크기가 큰 배치가 실패하면 더 작은 배치로 재시도하고 실제 배치 크기를 기록합니다. 장비나 CUDA/PyTorch 환경이 바뀌면 greedy 실행에서도 토큰 결과가 달라질 수 있으므로 비교 실험에는 동일한 런타임을 사용하세요.

## 로컬 Safety Judge

```bash
CUDA_VISIBLE_DEVICES=0 python -m lyra.experiment1.judge \
  --data-dir "$DATA_DIR" --run-dir "$RUN_DIR" --output-dir "$JUDGE_DIR"
```

Judge는 원본 이미지, 원본 질문, 실제 Target Prompt, 응답을 함께 받습니다. 따라서 Lyra-style에서 원본 질문을 볼 수 있는 모델은 Judge입니다. Shieldstral에는 위험성·안전상 거부·실질적 안전 경고를 독립된 yes/no 질문으로 제시합니다. `judgments.jsonl`의 `native_outputs`는 실제 최고 점수 토큰 및 yes/no 로짓을, `scores`는 두 응답 후보 기준 정규화된 yes 점수를, `flags`는 임계값 적용 결과를 기록합니다. 세 플래그는 동시에 참일 수 있습니다. 기본 임계값은 0.5이며 `--threshold`를 지원합니다. [Shieldstral 모델 카드](https://huggingface.co/mistralai/Shieldstral-1.0-3B)

VLSBench의 공식 세 범주 `Safe with Refusal`, `Safe with Warning`, `Unsafe`를 담는 `benchmark_label`은 Shieldstral에서 비워 둡니다. `shield_safe`는 로컬 위험성 질의가 임계값을 넘지 않은 비율입니다. 공식 평가에는 GPT-4o가 사용됐으며, 이 파이프라인은 Shieldstral의 로컬 분류 결과를 보고합니다. [VLSBench 평가 기준](https://ai45lab.github.io/VLSBench/)

수동 검토 결과는 `case_id,label` 열이 있는 UTF-8 CSV에 추가할 수 있습니다. `label`은 위 세 범주 중 하나이고 선택적으로 `run_id` 열을 추가할 수 있습니다. 분석에서는 수동 분류 비율과, 수동으로 평가한 사례에 대한 Shieldstral 위험성 판정의 일치율을 별도로 표시합니다.

## 전체 결과 분석

```bash
python -m lyra.experiment1.analysis \
  --run-dir "$RUN_DIR" --judge-dir "$JUDGE_DIR" --output-dir "$REPORT_DIR" --plot
```

판정 전에 생성 현황만 보려면 `--judge-dir`과 `--plot`을 생략합니다. 수동 라벨이 준비되면 `--manual-csv`에 파일 경로를 전달합니다.

| 산출물 | 내용 |
| --- | --- |
| `summary.csv` | 조건별, 조건×장르별, 조건×카테고리별, 조건×장르×카테고리별 집계와 비율 |
| `paired_summary.csv` | 같은 샘플·장르로 매칭한 Original↔각 창작 조건, Lyra-style↔Instruction-aware 간 비율 차이(%p) |
| `paired_cases.csv` | 대응 케이스 ID, 공백 정규화 후 응답 텍스트의 동일 여부, 양쪽 문자 수 |
| `issues.csv` | 추론 오류, 미시도·응답 누락, 판정 오류 및 미판정 현황 |
| `condition_flags.png` | `--plot` 사용 시 조건별 로컬 판정 비율 |

각 비율 열은 사용된 표본 수 `*_n`을 함께 기록하고, 분모가 0이면 비율을 비워 둡니다. 대응 비교는 해당 지표가 양쪽에 존재하는 샘플을 사용합니다. 거부율은 조건별 응답 특성입니다. Original은 질문 응답이고 나머지는 창작 과제이므로, 거부율 차이에서 jailbreak 성공률을 산출하지 않습니다.

## 코드 게시

```bash
git status --short --branch
git add -- .gitignore README.md docs/experiment1.md requirements-experiment1.txt lyra/experiment1/
git diff --cached --stat
git commit --only -m "Add VLSBench artistic genre experiment" -- .gitignore README.md docs/experiment1.md requirements-experiment1.txt lyra/experiment1/ &&
git push origin HEAD:main
```
