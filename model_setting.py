import os
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from .model import PretrainedModel, FineTuningModel
from .preprocessing import FineTuningDataset, DriftFineTuningDataset
from .utility.dataset_processor_tld import wrap_tld, to_esld


# 체크포인트마다 어휘 크기, 시퀀스 길이, 위치 인코딩 방식, 전처리가 모두 다르다.
# 가중치만 바꿔 끼우면 shape mismatch 로 즉시 터지므로 한 묶음으로 관리한다.
VARIANTS = {
    # DSN 2026 DRIFT 체크포인트. 공개되어 있어 서버에 바로 올릴 수 있다.
    'drift': {
        'subword_vocab': 30522,
        'subword_len': 30,
        'char_vocab': 43,
        'char_len': 77,
        'learned_pe': True,
        # 분류 헤드가 [MaxPool ; MeanPool] 을 이어 붙여 경로당 512, 두 경로 합쳐 1024.
        'clf_norm': 'pool',
        # TLD 를 떼고 eSLD 만 넣는다. 학습 데이터가 그 형태다.
        'preprocess': 'esld',
        'dataset_cls': DriftFineTuningDataset,
        'model_file': 'finetuning_0331_2010.pt',
        'tokenizer_file': 'tokenizer-0-30522-both.json',
    },
    # SURF 의 TLD-aware 변형. 가중치가 아직 확보되지 않았다.
    'surf-tld': {
        'subword_vocab': 32393,
        'subword_len': 35,
        'char_vocab': 2273,
        'char_len': 82,
        'learned_pe': False,
        'clf_norm': 'cls',
        # TLD 를 [.co][.kr] 로 감싸 어휘에 포함시킨다.
        'preprocess': 'wrap_tld',
        'dataset_cls': FineTuningDataset,
        'model_file': 'finetuning_0120_1528.pt',
        'tokenizer_file': 'tokenizer-2-32393-both-tld.json',
    },
}

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / 'artifacts'


class DomainClassifier:
    """도메인 문자열 하나로 DGA 여부를 판별한다.

    경로를 하드코딩하지 않는다. 서버마다 아티팩트 위치가 다르고, 가중치가
    저장소에 들어 있지 않아 배포 시점에 주입해야 하기 때문이다.

    환경변수
        SURF_MODEL_VARIANT   drift | surf-tld   (기본 drift)
        SURF_ARTIFACT_DIR    가중치와 토크나이저를 찾을 디렉터리
        SURF_MODEL_PATH      가중치 파일을 직접 지정할 때
        SURF_TOKENIZER_PATH  토크나이저 파일을 직접 지정할 때
    """

    def __init__(self, variant=None, model_path=None, tokenizer_path=None, device=None):
        variant = variant or os.getenv('SURF_MODEL_VARIANT', 'drift')
        if variant not in VARIANTS:
            raise ValueError(
                f"알 수 없는 variant: {variant!r}. 가능한 값: {sorted(VARIANTS)}"
            )

        cfg = VARIANTS[variant]
        self.variant = variant
        self.cfg = cfg
        self.device = torch.device(device or os.getenv('SURF_DEVICE', 'cpu'))

        artifact_dir = Path(os.getenv('SURF_ARTIFACT_DIR', DEFAULT_ARTIFACT_DIR))
        model_path = Path(
            model_path or os.getenv('SURF_MODEL_PATH') or artifact_dir / cfg['model_file']
        )
        tokenizer_path = Path(
            tokenizer_path or os.getenv('SURF_TOKENIZER_PATH')
            or artifact_dir / cfg['tokenizer_file']
        )

        for label, path in (('가중치', model_path), ('토크나이저', tokenizer_path)):
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} 파일이 없습니다: {path}\n"
                    f"SURF_ARTIFACT_DIR 로 위치를 지정하거나 아티팩트를 내려받으세요."
                )

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))

        self.pt_model_c = PretrainedModel(
            cfg['char_vocab'], 256, 8, 768, 12, cfg['char_len'],
            learned_pe=cfg['learned_pe']
        )
        self.pt_model_t = PretrainedModel(
            cfg['subword_vocab'], 256, 8, 768, 12, cfg['subword_len'],
            learned_pe=cfg['learned_pe']
        )
        self.ft_model = FineTuningModel(
            self.pt_model_t, self.pt_model_c, clf_norm=cfg['clf_norm']
        )

        self.processor = cfg['dataset_cls'](
            df=None, tokenizer=self.tokenizer,
            max_len_t=cfg['subword_len'], max_len_c=cfg['char_len']
        )

        state = torch.load(model_path, map_location=self.device, weights_only=False)
        if not isinstance(state, dict) or 'state_dict' in state:
            state = state.get('state_dict', state)
        self.ft_model.load_state_dict(state)
        self.ft_model.to(self.device)
        self.ft_model.eval()

        print(f"✅ Model loaded [{variant}] from {model_path}", flush=True)

    def preprocess(self, domain):
        """모델에 넣기 직전의 문자열. variant 마다 TLD 처리가 갈린다."""
        domain = domain.lower().strip().rstrip('.')
        mode = self.cfg['preprocess']
        if mode == 'wrap_tld':
            return wrap_tld(domain)
        if mode == 'esld':
            return to_esld(domain)
        return domain

    def predict(self, domain):
        processed_domain = self.preprocess(domain)

        x_token = self.processor.domain_to_token(processed_domain)
        x_char = self.processor.domain_to_ids(processed_domain)

        x_token_tensor = torch.from_numpy(x_token).unsqueeze(0).to(torch.long).to(self.device)
        x_char_tensor = torch.from_numpy(x_char).unsqueeze(0).to(torch.long).to(self.device)

        with torch.no_grad():
            logits = self.ft_model(x_token_tensor, x_char_tensor)
            pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)

        return pred, probs


if __name__ == '__main__':
    classifier = DomainClassifier()
    for d in ("google.com", "ubuntu.com", "naver.com",
              "gdheklhhsspojpiqjkre.com", "kqjwhdnalsjdhfnq.net"):
        pred, probs = classifier.predict(d)
        print(f"{d:32s} -> {pred}  (DGA 확률 {probs[0][1].item() * 100:.2f}%)")
