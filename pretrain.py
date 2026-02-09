import argparse
import polars as pl
pl.Config.set_engine_affinity(engine="streaming")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import SubTaskDataset, SpecialIDs
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from model import PretrainedModel
from tqdm import tqdm
import datetime
import wandb
from utility.dataset import get_train_set_tld, get_val_set_tld
from utility.config import PretrainConfig
from utility.path import path_model, path_tokenizer
from make_tokenizer_tld import train


def log_artifact(run, path, name, type_="model"):
    if run is not None:
        artifact = wandb.Artifact(name=name, type=type_)
        artifact.add_file(path)
        run.log_artifact(artifact)


def train_char(cfg, args) :
    now_date = datetime.datetime.now().strftime('%m%d_%H%M')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = None
    if args.use_wandb:
        run = wandb.init(project=args.project_name, name=args.run_name, config=vars(cfg), tags=['valid'])

    tokenizer_path = path_tokenizer.joinpath((f"tokenizer-{cfg.min_freq_subword}-{cfg.vocab_size_subword}-both-tld.json"))
    if not tokenizer_path.exists():
        _, paths = get_train_set_tld()
        train(file_paths=paths,
            text_col="domain",
            vocab_size=cfg.vocab_size_subword,
            min_freq=cfg.min_freq_subword,
            use_bert_pretokenizer=True,
            save_path=tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    print(f"Loaded Tokenizer Vocab Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == cfg.vocab_size_subword, "Tokenizer vocab size does not match!"

    train_df, _ = get_train_set_tld()
    val_df = get_val_set_tld()

    train_dataset = SubTaskDataset(
        train_df,
        max_len=cfg.max_len_char,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'char'
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_dataset = SubTaskDataset(
        val_df,
        max_len=cfg.max_len_char,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'char'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = PretrainedModel(
        vocab_size=cfg.vocab_size_char,
        d_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        num_layers=cfg.num_layers,
        max_len=cfg.max_len_char,
        tov_norm=cfg.tov_norm,
    ).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    bce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()

    best_loss = float('inf')
    global_step = 0
    
    interval_loss_sum_total = 0 
    interval_loss_sum_mtp = 0
    interval_loss_sum_tpp = 0
    interval_loss_sum_tov = 0

    train_loop = tqdm(total=args.total_steps, desc="[Train]", bar_format='{l_bar}{r_bar}')

    while global_step < args.total_steps :

        total_mtp_loss = 0
        total_tpp_loss = 0
        total_tov_loss = 0

        for X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov in train_dataloader :
            if global_step >= args.total_steps :
                break

            X_mtp, Y_mtp = X_mtp.to(device), Y_mtp.to(device)
            X_tpp, Y_tpp = X_tpp.to(device), Y_tpp.to(device)
            X_tov, Y_tov = X_tov.to(device), Y_tov.to(device)

            # --- T1: MTP Loss 계산 ---
            logits_mtp = model(X_mtp, task_type='MTP')
            loss_mtp = ce(
                logits_mtp.view(-1, logits_mtp.size(-1)), # (B*L, V)
                Y_mtp.view(-1)                            # (B*L)
            )

            # --- T2: TPP Loss 계산 ---
            logits_tpp = model(X_tpp, task_type='TPP')
            loss_tpp = ce(
                logits_tpp.view(-1, logits_tpp.size(-1)), # (B*L, L)
                Y_tpp.view(-1)                             # (B*L)
            )

            # --- T3: TOV Loss 계산 ---
            logits_tov = model(X_tov, task_type='TOV')
            loss_tov = bce(logits_tov, Y_tov) # Logits: (B x 2), Labels: (B)

            L_total = loss_mtp + loss_tpp + loss_tov

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            global_step += 1
            train_loop.update(1)

            total_mtp_loss += loss_mtp.item()
            total_tpp_loss += loss_tpp.item()
            total_tov_loss += loss_tov.item()

            interval_loss_sum_total += L_total.item()
            interval_loss_sum_mtp += loss_mtp.item()
            interval_loss_sum_tpp += loss_tpp.item()
            interval_loss_sum_tov += loss_tov.item()
            
            # 인터벌 로깅
            if global_step % args.log_interval == 0:
                avg_total_interval = interval_loss_sum_total / args.log_interval
                avg_mtp_interval = interval_loss_sum_mtp / args.log_interval
                avg_tpp_interval = interval_loss_sum_tpp / args.log_interval
                avg_tov_interval = interval_loss_sum_tov / args.log_interval
                
                if args.use_wandb:
                    wandb.log({
                        "step/interval_total_loss": avg_total_interval,
                        "step/interval_mtp_loss": avg_mtp_interval,
                        "step/interval_tpp_loss": avg_tpp_interval,
                        "step/interval_tov_loss": avg_tov_interval,
                    })

                train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval:.4f}")

                interval_loss_sum_total = 0 
                interval_loss_sum_mtp = 0
                interval_loss_sum_tpp = 0
                interval_loss_sum_tov = 0

            if global_step % args.val_check_interval == 0:
                val_loss = validate(model, val_dataloader, device, cfg, args)
                train_loop.write(f"[char] step {global_step} val_loss={val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = path_model.joinpath(f"{now_date}_{cfg.save_path.replace('.pt', '')}_step_{global_step}.pt")
                    torch.save(model.state_dict(), save_path)
                    if args.use_wandb:
                        pass
                        # log_artifact(run, save_path, f"{args.mode}_{now_date}") # wandb artifact 저장 필요 시

            current_step = train_loop.n + 1
            with torch.no_grad() :
                avg_mtp = total_mtp_loss / current_step
                avg_tpp = total_tpp_loss / current_step
                avg_tov = total_tov_loss / current_step
                avg_total = (total_mtp_loss + total_tpp_loss + total_tov_loss) / current_step

            train_loop.set_postfix(dict(avg_total=f'{avg_total:.4f}',
                                    avg_mtp=f'{avg_mtp:.4f}',
                                    avg_tpp=f'{avg_tpp:.4f}',
                                    avg_tov=f'{avg_tov:.4f}'), refresh=False)
    if args.use_wandb :
        wandb.finish()


def train_subword(cfg, args) :
    now_date = datetime.datetime.now().strftime('%m%d_%H%M')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = None
    if args.use_wandb:
        run = wandb.init(project=args.project_name, name=args.run_name, config=vars(cfg), tags=['valid'])

    tokenizer_path = path_tokenizer.joinpath((f"tokenizer-{cfg.min_freq_subword}-{cfg.vocab_size_subword}-both-tld.json"))
    if not tokenizer_path.exists():
        _, paths = get_train_set_tld()
        train(file_paths=paths,
            text_col="domain",
            vocab_size=cfg.vocab_size_subword,
            min_freq=cfg.min_freq_subword,
            use_bert_pretokenizer=True,
            save_path=tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    print(f"Loaded Tokenizer Vocab Size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == cfg.vocab_size_subword, "Tokenizer vocab size does not match!"

    train_df, _ = get_train_set_tld()
    val_df = get_val_set_tld()

    train_dataset = SubTaskDataset(
        train_df,
        max_len=cfg.max_len_subword,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'subword'
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_dataset = SubTaskDataset(
        val_df,
        max_len=cfg.max_len_subword,
        tokenizer=tokenizer,
        mask_ratio=cfg.mask_ratio,
        type = 'subword'
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    model = PretrainedModel(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.nhead,
        dim_feedforward=cfg.dim_feedforward,
        num_layers=cfg.num_layers,
        max_len=cfg.max_len_subword,
        tov_norm=cfg.tov_norm,
    ).to(device)

    ce = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    bce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()

    best_loss = float('inf')
    global_step = 0
    
    interval_loss_sum_total = 0 
    interval_loss_sum_mtp = 0
    interval_loss_sum_tpp = 0
    interval_loss_sum_tov = 0

    train_loop = tqdm(total=args.total_steps, desc="[Train]", bar_format='{l_bar}{r_bar}')

    while global_step < args.total_steps :

        total_mtp_loss = 0
        total_tpp_loss = 0
        total_tov_loss = 0
        
        for X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov in train_dataloader :
            if global_step >= args.total_steps :
                break

            X_mtp, Y_mtp = X_mtp.to(device), Y_mtp.to(device)
            X_tpp, Y_tpp = X_tpp.to(device), Y_tpp.to(device)
            X_tov, Y_tov = X_tov.to(device), Y_tov.to(device)

            # --- T1: MTP Loss 계산 ---
            logits_mtp = model(X_mtp, task_type='MTP')
            loss_mtp = ce(
                logits_mtp.view(-1, logits_mtp.size(-1)), # (B*L, V)
                Y_mtp.view(-1)                            # (B*L)
            )

            # --- T2: TPP Loss 계산 ---
            logits_tpp = model(X_tpp, task_type='TPP')
            loss_tpp = ce(
                logits_tpp.view(-1, logits_tpp.size(-1)), # (B*L, L)
                Y_tpp.view(-1)                             # (B*L)
            )

            # --- T3: TOV Loss 계산 ---
            logits_tov = model(X_tov, task_type='TOV')
            loss_tov = bce(logits_tov, Y_tov) # Logits: (B x 2), Labels: (B)

            L_total = loss_mtp + loss_tpp + loss_tov

            optimizer.zero_grad()
            L_total.backward()
            optimizer.step()

            global_step += 1
            train_loop.update(1)

            total_mtp_loss += loss_mtp.item()
            total_tpp_loss += loss_tpp.item()
            total_tov_loss += loss_tov.item()

            interval_loss_sum_total += L_total.item()
            interval_loss_sum_mtp += loss_mtp.item()
            interval_loss_sum_tpp += loss_tpp.item()
            interval_loss_sum_tov += loss_tov.item()
            
            # 인터벌 로깅
            if global_step % args.log_interval == 0:
                avg_total_interval = interval_loss_sum_total / args.log_interval
                avg_mtp_interval = interval_loss_sum_mtp / args.log_interval
                avg_tpp_interval = interval_loss_sum_tpp / args.log_interval
                avg_tov_interval = interval_loss_sum_tov / args.log_interval
                
                if args.use_wandb:
                    wandb.log({
                        "step/interval_total_loss": avg_total_interval,
                        "step/interval_mtp_loss": avg_mtp_interval,
                        "step/interval_tpp_loss": avg_tpp_interval,
                        "step/interval_tov_loss": avg_tov_interval,
                    })

                train_loop.write(f"[Step {global_step} Interval Log]: Train Loss: {avg_total_interval:.4f}")

                interval_loss_sum_total = 0 
                interval_loss_sum_mtp = 0
                interval_loss_sum_tpp = 0
                interval_loss_sum_tov = 0

            if global_step % args.val_check_interval == 0:
                val_loss = validate(model, val_dataloader, device, cfg, args)
                train_loop.write(f"[subword] step {global_step} val_loss={val_loss:.4f}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = path_model.joinpath(f"{now_date}_{cfg.save_path.replace('.pt', '')}_step_{global_step}.pt")
                    torch.save(model.state_dict(), save_path)
                    if args.use_wandb:
                        pass
                        # log_artifact(run, save_path, f"{args.mode}_{now_date}") # wandb artifact 저장 필요 시

            current_step = train_loop.n + 1
            with torch.no_grad() :
                avg_mtp = total_mtp_loss / current_step
                avg_tpp = total_tpp_loss / current_step
                avg_tov = total_tov_loss / current_step
                avg_total = (total_mtp_loss + total_tpp_loss + total_tov_loss) / current_step

            train_loop.set_postfix(dict(avg_total=f'{avg_total:.4f}',
                                    avg_mtp=f'{avg_mtp:.4f}',
                                    avg_tpp=f'{avg_tpp:.4f}',
                                    avg_tov=f'{avg_tov:.4f}'), refresh=False)
    if args.use_wandb :
        wandb.finish()


def validate(model, dataloader, device, cfg, args):
    model.eval()
    total_loss = 0
    mtp_loss_total = 0
    tpp_loss_total = 0
    tov_loss_total = 0

    ce = torch.nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    bce = torch.nn.CrossEntropyLoss()

    val_loop = tqdm(dataloader, desc="Validation", bar_format='{l_bar}{r_bar}', leave=False)
    
    with torch.no_grad():
        for X_mtp, Y_mtp, X_tpp, Y_tpp, X_tov, Y_tov in val_loop :
            
            X_mtp, Y_mtp = X_mtp.to(device), Y_mtp.to(device)
            X_tpp, Y_tpp = X_tpp.to(device), Y_tpp.to(device)
            X_tov, Y_tov = X_tov.to(device), Y_tov.to(device)

            # --- T1: MTP Loss 계산 ---
            logits_mtp = model(X_mtp, task_type='MTP')
            loss_mtp = ce(
                logits_mtp.view(-1, logits_mtp.size(-1)), # (B*L, V)
                Y_mtp.view(-1)                            # (B*L)
            )

            # --- T2: TPP Loss 계산 ---
            logits_tpp = model(X_tpp, task_type='TPP')
            loss_tpp = ce(
                logits_tpp.view(-1, logits_tpp.size(-1)), # (B*L, L)
                Y_tpp.view(-1)                             # (B*L)
            )

            # --- T3: TOV Loss 계산 ---
            logits_tov = model(X_tov, task_type='TOV')
            loss_tov = bce(logits_tov, Y_tov) # Logits: (B x 2), Labels: (B)

            L_total = loss_mtp + loss_tpp + loss_tov

            total_loss += L_total.item()
            mtp_loss_total += loss_mtp.item()
            tpp_loss_total += loss_tpp.item()
            tov_loss_total += loss_tov.item()

            val_loop.update(1)

    avg_total = total_loss / len(dataloader)
    avg_mtp = mtp_loss_total / len(dataloader)
    avg_tpp = tpp_loss_total / len(dataloader)
    avg_tov = tov_loss_total / len(dataloader)

    if args.use_wandb:
        wandb.log({
            "step/val_total_loss": avg_total,
            "step/val_mtp_loss": avg_mtp,
            "step/val_tpp_loss": avg_tpp,
            "step/val_tov_loss": avg_tov,
        })
    
    model.train()

    return avg_total


def main() :
   parser = argparse.ArgumentParser()
   cfg = PretrainConfig()
   parser.add_argument("--mode", choices=["char", "subword"], required=True,
                        help="Pre-training mode: char or subword")
   parser.add_argument("--save", type=str, default="pretrained.pt", help="Path to save model state dict")
   parser.add_argument("--total_steps", type=int, default=10000000, help="Total training steps")
   parser.add_argument("--val_check_interval", type=int, default=20000, help="Steps between validation")
   parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
   parser.add_argument("--log_interval", type=int, default=1000, help="Steps between logging")
   parser.add_argument("--project_name", type=str, default="dga-pretrain", help="Wandb project name")
   parser.add_argument("--run_name", type=str, default="run", help="Wandb run name")
   parser.add_argument("--tov_norm", type=str, choices=["cls", "pool"], default="cls", help="TOV pooling strategy")
   parser.add_argument("--tokenizer_min_freq", type=int, default=cfg.min_freq_subword, help="Tokenizer min frequency")
   parser.add_argument("--tokenizer_vocab_size", type=int, default=cfg.vocab_size_subword, help="Tokenizer vocab size")

   args = parser.parse_args()
   args.use_wandb = not args.no_wandb

   cfg = PretrainConfig(
        save_path=args.save, 
        tov_norm=args.tov_norm,
        min_freq_subword=args.tokenizer_min_freq,
        vocab_size_subword=args.tokenizer_vocab_size
    )

   if args.mode == "char" :
       train_char(cfg, args)
   elif args.mode == "subword" :
       train_subword(cfg, args)

if __name__ == '__main__':
    main()