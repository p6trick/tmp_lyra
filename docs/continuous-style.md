# DGX Spark에서 StyleVector · EmoStyle 실행

## DGX에서 Lyra 코드 가져오기

로컬 변경사항을 GitHub `main` 브랜치에 올린 뒤 DGX에서 해당 버전을 가져옵니다. DGX에 Lyra 체크아웃이 이미 있다면 아래 `LYRA_REPO`를 그 위치로 지정합니다. 예시는 `/ds2home/Lyra`를 사용합니다.

```bash
export LYRA_REPO=/ds2home/Lyra
cd "$LYRA_REPO" && git pull --ff-only origin main
```

DGX에서 처음 설치한다면 원하는 체크아웃 경로로 클론합니다.

```bash
export LYRA_REPO=/ds2home/Lyra
git clone --branch main https://github.com/p6trick/tmp_lyra.git "$LYRA_REPO" && cd "$LYRA_REPO"
```

이후 명령은 이 Lyra 저장소 루트에서 실행합니다. `LYRA_ASSETS`에는 체크포인트, Hugging Face 캐시, 공식 EmoStyle 체크아웃을 둘 쓰기 가능한 별도 경로를 지정합니다. 예시의 `/ds2home/lyra-assets`도 변경할 수 있습니다.

```bash
export LYRA_ASSETS=/ds2home/lyra-assets
export HF_HOME="$LYRA_ASSETS/hf-cache"
export EMOSTYLE_REPO="$LYRA_ASSETS/EmoStyle-src"
mkdir -p "$LYRA_ASSETS"
```

## Conda 환경과 패키지

두 환경 모두 Python 3.11과 PyTorch 2.13.0 `cu130`을 사용합니다. 이미지 환경에는 같은 CUDA 빌드의 TorchVision 0.28.0을 설치합니다. PyTorch 인덱스에서 CPython 3.11용 Linux aarch64 wheel을 확인했습니다. 패키지는 Conda 환경 내부에 설치합니다. [PyTorch 설치 목록](https://pytorch.org/get-started/previous-versions/) · [Torch ARM64 wheel](https://download.pytorch.org/whl/cu130/torch/) · [TorchVision ARM64 wheel](https://download.pytorch.org/whl/cu130/torchvision/)

```bash
conda env create -f environments/lyra-text.yml
conda activate lyra-text
python -m pip install --only-binary=:all: 'torch==2.13.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install --only-binary=:all: -r environments/lyra-text.txt
python -m pip check

conda env create -f environments/lyra-image.yml
conda activate lyra-image
python -m pip install --only-binary=:all: 'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install --only-binary=:all: -r environments/lyra-image.txt
python -m pip check
```

EmoStyle 공식 설치 예시는 Python 3.9와 PyTorch 2.4.0 `cu124`를 제시합니다. 공개 추론 소스의 `str | None` 타입 표기에는 Python 3.10 이상이 필요합니다. Lyra 환경은 Python 3.11과 ARM64용 `cu130` wheel을 사용하고, 공식 추론 경로가 import하는 패키지로 설치 범위를 제한합니다. DeepSpeed 등 학습 의존성은 설치 대상에서 제외됩니다. [EmoStyle 공식 설치 가이드](https://github.com/JingyuanYY/EmoStyle#requirements) · [SentencePiece ARM64 wheel](https://pypi.org/project/sentencepiece/0.2.1/#files)

## Text · StyleVector

```bash
conda activate lyra-text
hf download Qwen/Qwen3-8B --local-dir "$LYRA_ASSETS/Qwen3-8B"

bash scripts/continuous_style/stylevector.sh \
  --model-path "$LYRA_ASSETS/Qwen3-8B"
```

Qwen3-8B snapshot에는 `config.json`, `generation_config.json`, tokenizer 파일, `model.safetensors.index.json` 및 이에 대응하는 모든 weight shard가 있어야 합니다. 스크립트는 해당 디렉터리에서 로컬 파일을 읽습니다. 기본값은 `cuda:0`, BF16, zero-based decoder layer `17`, steering `alpha=1.0`입니다. `--layer`, `--alpha`, `--config`, `--device`, `--output-dir`로 지정할 수 있습니다. [Qwen3-8B 모델 카드](https://huggingface.co/Qwen/Qwen3-8B)

ACL 논문과 arXiv에서 저자가 연결한 공식 구현 저장소를 확인하지 못했습니다. Lyra 구현은 논문 §3.2의 평균 차이를 따릅니다. 각 예제에서 동일한 프롬프트에 긍정·중립 응답을 각각 연결하고, 선택한 블록 출력에서 마지막 응답 토큰의 activation을 추출합니다. `mean(positive - negative)`를 정규화 없이 FP32로 저장하고, 추론 시 해당 블록 출력에 `alpha * vector`를 더합니다. 논문은 Llama-2-7B-chat, 사용자 글 이력, LLM이 생성한 중립 응답, 검증셋 기반 레이어·강도 선택을 사용합니다. 이 테스트는 Qwen3-8B, 손으로 작성한 대조 쌍 3개, 고정 기본 레이어, thinking 비활성화 및 greedy decoding을 사용합니다. 논문 §3.3의 프롬프트 이후 위치 조건에 따라 프롬프트 prefill은 그대로 통과시키고 KV 캐시를 사용한 이후 decoder 호출에 벡터를 적용합니다. 첫 응답 토큰은 프롬프트에서 예측됩니다. [StyleVector 논문](https://aclanthology.org/2025.acl-long.353.pdf) · [arXiv](https://arxiv.org/abs/2503.05213)

기본 출력은 `runs/continuous_style/stylevector/<run-id>/` 아래에 생깁니다.

| 파일 | 내용 |
| --- | --- |
| `baseline.txt` | 동일 프롬프트의 `alpha=0` 응답 |
| `steered.txt` | 동일 프롬프트의 `alpha=1` 응답 |
| `style_vector.pt` | CPU FP32 벡터 |
| `run.json` | 모델 및 환경 정보, 프롬프트·대조 쌍, 레이어·강도, 벡터 크기와 hook 통계 |
| `inference.log` | 모델 로딩, 추출 및 생성 로그 |

성공한 실행에서 `steered_decode_steps`와 `first_decode_max_abs_change`는 양수입니다. Greedy decoding 결과 두 응답 문자열이 같을 수 있습니다.

## Image · EmoStyle

공식 소스를 지정한 리비전으로 가져온 다음 경로 패치를 적용합니다. 패치는 FLUX, VAE, USO projector, SigLIP의 로컬 경로를 환경 변수로 읽게 합니다. CLIP과 T5를 구분할 때는 체크포인트 디렉터리의 마지막 이름만 사용하도록 수정합니다. 래퍼가 환경 변수를 전달하고 공식 `inference_emostyle.py`를 호출합니다. [고정한 EmoStyle 소스](https://github.com/JingyuanYY/EmoStyle/tree/00cfe4cf9d329121ac88cbe6d5622723846d0919)

```bash
git clone https://github.com/JingyuanYY/EmoStyle.git "$EMOSTYLE_REPO"
git -C "$EMOSTYLE_REPO" checkout --detach 00cfe4cf9d329121ac88cbe6d5622723846d0919
git -C "$EMOSTYLE_REPO" apply "$PWD/scripts/continuous_style/emostyle-local-paths.patch"
```

Hugging Face에서 FLUX.1-dev 접근 조건에 동의한 계정으로 로그인하고 필요한 파일을 받습니다. `sadness.bin`은 공식 `--mapper_ckpt`에 전달합니다. 공식 추론 CLI는 mapper 체크포인트 하나를 받으므로 다운로드 예시는 `sadness.bin`을 지정합니다. [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) · [EmoStyle 체크포인트](https://huggingface.co/lilwhit3/EmoStyle/tree/main/StyleDictionary) · [USO 체크포인트](https://huggingface.co/bytedance-research/USO/tree/main/uso_flux_v1.0) · [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384/tree/main)

```bash
conda activate lyra-image
hf auth login
hf download black-forest-labs/FLUX.1-dev \
  --local-dir "$LYRA_ASSETS/FLUX.1-dev" \
  --include flux1-dev.safetensors ae.safetensors \
    'text_encoder/*' 'text_encoder_2/*' 'tokenizer/*' 'tokenizer_2/*'
cp "$LYRA_ASSETS/FLUX.1-dev/tokenizer/"* "$LYRA_ASSETS/FLUX.1-dev/text_encoder/"
cp "$LYRA_ASSETS/FLUX.1-dev/tokenizer_2/"* "$LYRA_ASSETS/FLUX.1-dev/text_encoder_2/"
ln -sfnT text_encoder "$LYRA_ASSETS/FLUX.1-dev/clip-vit-l14"

hf download bytedance-research/USO \
  uso_flux_v1.0/dit_lora.safetensors uso_flux_v1.0/projector.safetensors \
  --local-dir "$LYRA_ASSETS/USO"
hf download google/siglip-so400m-patch14-384 \
  config.json preprocessor_config.json model.safetensors \
  --local-dir "$LYRA_ASSETS/siglip"
hf download lilwhit3/EmoStyle StyleDictionary/sadness.bin \
  --local-dir "$LYRA_ASSETS/EmoStyle"

bash scripts/continuous_style/emostyle.sh \
  --emostyle-repo "$EMOSTYLE_REPO" \
  --models-dir "$LYRA_ASSETS"
```

기본 입력은 `configs/continuous_style/emostyle-example.png`입니다. `--image`로 다른 이미지 1장을 지정할 수 있습니다. 기본값은 `sadness`, `emo_strength=1.0`, 1024×1024, 25 steps, `guidance=45`, `seed=126`입니다. `--config`에서 모델 파일 경로와 추론 기본값을 바꿀 수 있습니다. 경로는 절대 경로 또는 `--models-dir` 기준이며 CLIP 디렉터리 이름에는 `clip`이 필요합니다. `--emotion`을 바꾸면 해당 emotion용 mapper 체크포인트를 준비합니다.

공식 소스에서 `EMOTION2ID`의 `sadness`는 7입니다. `emo_embeds`는 이 인덱스의 `(1, 8)` one-hot 입력이며, mapper는 이를 SigLIP 콘텐츠 토큰과 함께 받아 `(1, 192, 3072)` 스타일 임베딩을 반환합니다. CLI의 `--emo_strength`는 mapper 출력에 곱해지며, Lyra에서는 `--emo-strength`로 전달합니다. FLUX `guidance`는 별도 인자입니다. 임베딩 텐서는 공식 추론 코드의 변수에서 접근할 수 있습니다. [공식 추론 소스](https://github.com/JingyuanYY/EmoStyle/blob/00cfe4cf9d329121ac88cbe6d5622723846d0919/inference_emostyle.py)

기본 출력은 `runs/continuous_style/emostyle/<run-id>/`에 생깁니다. `stylized.png`에 결과 1장을, `run.json`에 입력·체크포인트 경로와 실행 설정을, `inference.log`에 공식 CLI의 stdout과 stderr를 저장합니다. `--output-dir`로 상위 출력 경로를 바꿀 수 있습니다.

## DGX에서 확인할 항목

공식 배포 파일과 소스에서 버전 및 경로를 확인했습니다. 지정한 DGX Spark에서 패키지 설치, 체크포인트 로딩 및 GPU 추론은 미검증 상태입니다. 환경마다 `uname -m`, `nvidia-smi`, 아래 명령으로 ARM64, CUDA runtime `13.0`, GB10 compute capability `(12, 1)` 및 실제 BF16 CUDA 연산을 확인할 수 있습니다. R580은 NVIDIA가 명시한 CUDA 13.x 드라이버 요구 범위에 들어갑니다. [NVIDIA CUDA 호환성](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

```bash
python - <<'PY'
import torch

print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
print((x @ x)[0, 0].item())
PY
```

- EmoStyle에서 BF16 행렬 연산과 FLUX attention은 서로 다른 커널을 사용합니다. 간단한 CUDA 검사 성공 후에도 이미지 추론이 실패하면 `inference.log`에서 실패한 연산을 확인합니다. `--offload`로 공식 CPU offload를 선택할 수 있으며, Spark unified memory에서 효과는 실제 실행으로 확인해야 합니다.
- `401`이나 `403`이면 FLUX 모델 접근 동의와 `hf auth login` 계정을 확인합니다. `sadness.bin`과 고정한 코드 리비전의 전체 mapper state dict 호환성은 DGX에서 첫 로딩 시 확인됩니다.
- 두 래퍼는 기본 출력 디렉터리를 Git에서 제외합니다. `--output-dir`을 바꿀 때에도 대용량 결과 및 체크포인트는 저장소 외부 경로에 두면 됩니다.
