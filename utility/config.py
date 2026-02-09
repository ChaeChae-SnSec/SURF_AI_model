from dataclasses import dataclass, field
from typing import Optional
import datetime
import sys, os

@dataclass
class DatasetConfig:
    train_size = 1_500_000
    val_size = 150_000
    # The rest is for test size
    

@dataclass
class PretrainConfig:
    # Data
    max_len_char: int = 82 # 77 + 5 (TLD 포함함에 따라 5 늘림)
    vocab_size_char: int = 2273 # 44 + 4403 -> 45 + 1869
    text_col: str = "domain"
    label_col: str = "label"
    
    # Vocabulary (Subword)
    max_len_subword: int = 35 # 30
    vocab_size_subword: int = 32393
    min_freq_subword: int = 2
    use_bert_pretokenizer: bool = False
    
    # Model
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 768
    dropout: float = 0.1
    
    # Training
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 1e-4
    
    # Pretraining Tasks
    mask_ratio: float = 0.15
    shuffle_prob: float = 0.5
    tov_norm: str = "cls"  # "cls" or "pool"
    ignore_index: int = -100
    
    # Output
    save_path: str = "pretrained.pt"


def get_wandb_mode():
    # VS Code에서 Run 버튼으로 실행된 경우
    if "VSCODE_PID" in os.environ:
        return "disabled"
    return "online"

@dataclass
class FinetuningConfig:
    token_weights_path: str = '0110_1545_pretrained_step_2080000.pt'
    char_weights_path: str = '0110_1541_char_step_1140000.pt'
    tokenizer_path: str = "tokenizer-2-32393-both-tld.json"

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 768
    max_len_token: int = 35 # 30
    max_len_char: int = 82 # 77
    vocab_size_token: int = 32393 # tokenizer_m.vocab_size = 30522 + 4403 + 1
    vocab_size_char: int = 2273

    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    backbone_lr: float = 1e-6
    num_workers: int = 4
    log_interval_steps: int = 1000
    unfreeze_at_epoch: float = 0.5

    # Ablation Study / Training Strategy
    use_token: bool = True       # Token Backbone 사용 여부
    use_char: bool = True        # Char Backbone 사용 여부
    freeze_backbone: bool = True # "시작"할 때 Backbone 고정 여부 (True면 Head만 학습)
    clf_norm: str = "cls"       # "cls" or "pool"

    # Logging & Project
    project_name: str = 'drift-finetune'
    wandb_mode: str = field(default_factory=get_wandb_mode)
    run_name_prefix: str = 'finetuning'
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime('%m%d_%H%M'))

    @property
    def best_filename(self) -> str:
        return f"{self.run_name_prefix}_{self.timestamp}"