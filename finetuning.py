import pandas as pd
import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import FineTuningDataset
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from model import PretrainedModel, FineTuningModel
from tqdm import tqdm
import time
import datetime
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
from utility import dataset
from utility.path import path_tokenizer, path_model
from utility.config import FinetuningConfig
import argparse

def load_pretrain_weights(pt_model, weigths_path, device) :
    state_dict = torch.load(weigths_path, map_location=device)

    weights_to_load = {}

    for name, param in state_dict.items() :
        if name.startswith('transformer.') or name.startswith('embedding.') or name.startswith('positional_encoding.') :
            weights_to_load[name] = param

    pt_model.load_state_dict(weights_to_load, strict=False)
    return pt_model

def fine_tune_dga_classifier(pt_model_t, pt_model_c,
                             train_dataloader, val_dataloader, weights_path_t, weights_path_c,
                             device, num_epochs, log_interval_steps, save_path,
                             use_token=True, use_char=True, freeze_backbone=True,
                             unfreeze_at_epoch=0.5, clf_norm='cls',
                             learning_rate=1e-4, backbone_lr=1e-6,):

    pt_t = load_pretrain_weights(pt_model_t, weights_path_t, device) if use_token else None
    pt_c = load_pretrain_weights(pt_model_c, weights_path_c, device) if use_char else None
    
    ft_model = FineTuningModel(
        pretrain_model_t=pt_t, 
        pretrain_model_c=pt_c, 
        freeze_backbone=freeze_backbone,
        clf_norm=clf_norm # or 'cls' method
    ).to(device)

    param_groups = [
        {
            'params': ft_model.classifier_head.parameters(), 
            'lr': learning_rate # 로깅 시 index가 0인 것이 로깅되기 때문에 헤드 파라미터를 먼저 append해주어야함
        }
    ]
    
    if ft_model.use_token:
        param_groups.extend([
            {'params': ft_model.transformer_encoder_t.parameters(), 'lr': backbone_lr},
            {'params': ft_model.embedding_t.parameters(), 'lr': backbone_lr},
            {'params': ft_model.positional_encoding_t.parameters(), 'lr': backbone_lr}
        ])
    if ft_model.use_char:
        param_groups.extend([
            {'params': ft_model.transformer_encoder_c.parameters(), 'lr': backbone_lr},
            {'params': ft_model.embedding_c.parameters(), 'lr': backbone_lr},
            {'params': ft_model.positional_encoding_c.parameters(), 'lr': backbone_lr}
        ])

    optimizer = optim.Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    if freeze_backbone:
        unfreeze_step = int(len(train_dataloader) * unfreeze_at_epoch) if unfreeze_at_epoch is not None else float('inf')
        backbone_unfrozen = False # 추후 백본 해제 시 True로 변경됨
    else: # freeze_backbone=False 이면 full finetuning하므로
        unfreeze_step = None
        backbone_unfrozen = True
    best_val_loss = float('inf')
    global_step = 0 # 전체 스텝 카운터
    
    interval_loss_sum_total = 0 # 로깅 인터벌 동안의 손실 누적합
    interval_batch_counter = 0 # 로깅 인터벌 동안의 배치 카운터

    for epoch in range(num_epochs) :
        ft_model.train()
        total_loss = 0
        train_loop = tqdm(train_dataloader, desc=f'FineTune Epoch {epoch+1}', 
                        bar_format="{l_bar}{n_fmt}/{total_fmt} | [{elapsed}<{remaining} {postfix}]",
                        leave=False)

        for X_token, X_char, y_train in train_loop :
            global_step += 1

            if freeze_backbone and not backbone_unfrozen and global_step >= unfreeze_step:
                ft_model.set_backbone_freezing(freeze=False) # 모든 파라미터 해제
                backbone_unfrozen = True
                print(f"--- [Step {global_step}] Backbone Unfrozen. Momentum Preserved. ---")

            X_token, X_char, y_train = X_token.to(device), X_char.to(device), y_train.to(device)
            optimizer.zero_grad()

            logits = ft_model(
                X_token if use_token else None, 
                X_char if use_char else None
            )
            
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']

            total_loss += loss.item()

            interval_loss_sum_total += loss.item()
            interval_batch_counter += 1

            if interval_batch_counter == log_interval_steps :

                avg_total_interval_loss = interval_loss_sum_total / interval_batch_counter


                if global_step % 10000 == 0 :

                    avg_val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_finetuning(ft_model, val_dataloader, device)

                    wandb.log({
                        'train/loss' : avg_total_interval_loss,
                        'val/loss': avg_val_loss,
                        'val/acc' : val_acc,
                        'val/prec': val_precision,
                        'val/recall' : val_recall,
                        'val/f1': val_f1,
                        'train/lr': current_lr
                    }, step=global_step)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(ft_model.state_dict(), save_path)

                    train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f},"
                        f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")
                else :
                    wandb.log({
                        'train/loss' : avg_total_interval_loss,
                        'train/lr': current_lr
                    }, step=global_step)

                interval_loss_sum_total = 0 
                interval_batch_counter = 0

            current_step = train_loop.n + 1
            avg_total = total_loss / current_step

            train_loop.set_postfix(avg_loss=f'{avg_total:.4f}', refresh=False)

def evaluate_finetuning(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    val_preds = []
    val_labels = []

    val_loop = tqdm(dataloader, desc="Validation", disable=True,
                    bar_format="{l_bar}{n_fmt}/{total_fmt} | [{elapsed}<{remaining} {postfix}]",
                    leave=False)

    with torch.no_grad():
        for X_token, X_char, y_val in val_loop:
            X_token, X_char, y_val = X_token.to(device), X_char.to(device), y_val.to(device)
            
            logits = model(
                X_token if model.use_token else None, 
                X_char if model.use_char else None
            )
            
            loss = criterion(logits, y_val)
            total_loss += loss.item()

            predicted_labels = torch.argmax(logits, dim=1)

            val_preds.append(predicted_labels.cpu())
            val_labels.append(y_val.cpu())

            current_step = val_loop.n + 1
            avg_total = total_loss / current_step

            val_loop.set_postfix(avg_loss=f'{avg_total:.4f}', refresh=False)

    avg_loss = total_loss / len(dataloader)

    val_preds = torch.cat(val_preds)
    val_labels = torch.cat(val_labels)

    accuracy = (val_preds == val_labels).sum().item() / len(val_labels)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds, zero_division=0)

    model.train()
    return avg_loss, accuracy, precision, recall, f1

def main():
    cfg = FinetuningConfig()

    parser = argparse.ArgumentParser(description="Fine-tuning DGA Classifier")

    # 경로
    parser.add_argument("--tokenizer_path", type=str, default=cfg.tokenizer_path)
    parser.add_argument("--use_bert_pretokenizer", type=bool, default=False)
    parser.add_argument("--project_name", type=str, default=cfg.project_name)
    parser.add_argument("--best_filename", type=str, default=cfg.best_filename)
    parser.add_argument("--wandb_mode", type=str, default=cfg.wandb_mode)
    parser.add_argument("--timestamp", type=str, default=cfg.timestamp)

    # 모델 구조 관련
    parser.add_argument("--d_model", type=int, default=cfg.d_model)
    parser.add_argument("--nhead", type=int, default=cfg.nhead)
    parser.add_argument("--dim_feedforward", type=int, default=cfg.dim_feedforward)
    parser.add_argument("--num_layers", type=int, default=cfg.num_layers)
    parser.add_argument("--max_len_token", type=int, default=cfg.max_len_token)
    parser.add_argument("--max_len_char", type=int, default=cfg.max_len_char)
    parser.add_argument("--vocab_size_char", type=int, default=cfg.vocab_size_char)

    # 학습 하이퍼파라미터
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=cfg.num_workers)
    parser.add_argument("--num_epochs", type=int, default=cfg.num_epochs)
    parser.add_argument("--learning_rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--backbone_lr", type=float, default=cfg.backbone_lr)
    parser.add_argument("--log_interval_steps", type=int, default=cfg.log_interval_steps)

    # 플래그
    parser.add_argument("--use_token", default=cfg.use_token)
    parser.add_argument("--use_char", default=cfg.use_char)
    parser.add_argument("--freeze_backbone", default=cfg.freeze_backbone)
    parser.add_argument("--clf_norm", type=str, default=cfg.clf_norm, choices=['cls', 'pool'])

    # 가중치
    parser.add_argument("--token_weights_path", type=str, default=cfg.token_weights_path)
    parser.add_argument("--char_weights_path", type=str, default=cfg.char_weights_path)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 토크나이저 & 경로 & 완디비
    if args.use_bert_pretokenizer :
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    else :
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path_tokenizer.joinpath(args.tokenizer_path)))
    vocab_size_token = tokenizer.vocab_size

    save_dir = path_model.joinpath(args.timestamp)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir.joinpath(f"{args.best_filename}.pt")

    wandb.init(project=args.project_name, name=args.best_filename, 
               config=vars(args), mode=args.wandb_mode, tags=['valid'])
               
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    # 데이터셋
    train_df, _ = dataset.get_train_set_tld()
    val_df = dataset.get_val_set_tld()

    train_dataset = FineTuningDataset(train_df, tokenizer=tokenizer, 
                                      max_len_t=args.max_len_token, max_len_c=args.max_len_char)
    val_dataset = FineTuningDataset(val_df, tokenizer=tokenizer, 
                                    max_len_t=args.max_len_token, max_len_c=args.max_len_char)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers)

    pt_model_t = PretrainedModel(vocab_size=vocab_size_token, d_model=args.d_model, 
                                 n_heads=args.nhead, dim_feedforward=args.dim_feedforward, 
                                 num_layers=args.num_layers, max_len=args.max_len_token)
    pt_model_c = PretrainedModel(vocab_size=args.vocab_size_char, d_model=args.d_model, 
                                 n_heads=args.nhead, dim_feedforward=args.dim_feedforward, 
                                 num_layers=args.num_layers, max_len=args.max_len_char)
    
    # 학습 실행
    fine_tune_dga_classifier(
        pt_model_t,
        pt_model_c,
        train_dataloader,
        val_dataloader,
        weights_path_t=path_model.joinpath(args.token_weights_path),
        weights_path_c=path_model.joinpath(args.char_weights_path),
        device=device,
        num_epochs=args.num_epochs,
        log_interval_steps=args.log_interval_steps,
        save_path=best_model_path,
        use_token=args.use_token,
        use_char=args.use_char,
        freeze_backbone=args.freeze_backbone,
        clf_norm=args.clf_norm,
        learning_rate=args.learning_rate,
        backbone_lr=args.backbone_lr
    )
    
    if best_model_path.exists():
        artifact = wandb.Artifact(name=args.best_filename, type="model")
        artifact.add_file(str(best_model_path))
        wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()