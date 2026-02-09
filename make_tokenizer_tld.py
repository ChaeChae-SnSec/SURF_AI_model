import polars as pl
import argparse
import glob
from pathlib import Path
from utility.config import PretrainConfig
from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers
)
from utility.dataset import get_train_set_tld

def get_corpus_batches(file_paths, column="domain", batch_size=10000):
    """
    Parquet 파일들을 Lazy하게 스캔하여 중복을 제거한 후,
    Tokenizer 학습을 위해 배치 단위로 yield 합니다."""

    q = pl.scan_parquet(file_paths)
    q = q.select(column).unique().drop_nulls()
    df_unique = q.collect()
    
    total_rows = df_unique.height

    for i in range(0, total_rows, batch_size):
        yield df_unique[column].slice(i, batch_size).to_list()

def train(file_paths,
        text_col: str,
        vocab_size: int,
        min_freq: int,
        save_path : str,
        freq_th: int = 10,
        use_bert_pretokenizer: bool = True
    ) -> Tokenizer:

    print("[.tld] 추출 중...: 빈도수 10 이상.")
    tld_df = (
        pl.scan_parquet(paths)
        .select(pl.col(text_col).str.extract_all(r"\[\.[a-zA-Z0-9-]+\]"))
        .explode(text_col)
        .drop_nulls()
        # TLD별로 그룹화하여 개수를 셉니다.
        .group_by(text_col)
        .len(name="count") 
        # 빈도가 높은 순서대로 정렬 (선택 사항)
        .sort("count", descending=True)
        .collect()
    )

    filtered_tld_df = tld_df.filter(pl.col("count") >= freq_th)
    output_csv_path = str(save_path)[:-5] + ".csv"
    filtered_tld_df.write_csv(output_csv_path)
    tld_special_tokens = filtered_tld_df[text_col].to_list()

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents()])

    if use_bert_pretokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # Token IDs: [PAD]:0, [UNK]:1, [CLS]:2, [SEP]:3, [MASK]:4, [TRUNC]:5, [SPARSE_TLD]:6, TLDs:7~ (순서대로 할당)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[TRUNC]", "[SPARSE_TLD]"] + tld_special_tokens
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, # 32393 = 30522(bert) + 2 + 1869:len(tld_special_tokens)
        min_frequency=min_freq,
        special_tokens=special_tokens)

    # training
    corpus_iter = get_corpus_batches(file_paths=file_paths, column=text_col)
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ("[PAD]", tokenizer.token_to_id("[PAD]")),
        ],
    )

    tokenizer.save(str(save_path))
    return tokenizer

if __name__ == "__main__":
    from transformers import PreTrainedTokenizerFast
    from utility.dataset import get_train_set_tld # 이거 get_train_set_pretrain -> get_train_set_tld로 바꿨는데 꼬옥 더블체크 부탁드립니다...
    from utility.config import PretrainConfig
    from utility.path import path_tokenizer

    cfg = PretrainConfig
    vocab = cfg.vocab_size_subword
    minfreq = cfg.min_freq_subword
    _, paths = get_train_set_tld()

    tokenizer_path = path_tokenizer.joinpath((f"tokenizer-{cfg.min_freq_subword}-{cfg.vocab_size_subword}-both-tld.json"))

    if not tokenizer_path.exists():
        trained_tokenizer = train(
            file_paths=paths,
            text_col="domain",
            vocab_size=vocab,
            min_freq=minfreq,
            save_path = tokenizer_path,
            freq_th=10,
            use_bert_pretokenizer=True
        )
    else:
        trained_tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))

    # # 새로운 wrap_tld 검증: [.com]이 하나의 토큰으로 나오는지 확인
    # test_domain = "getwhatyouwant[.co][.ru]"
    # encoded = trained_tokenizer(test_domain)
    # print(f"Test Sentence: {test_domain}")
    # print(f"Encoded: {encoded}\n")

    # tokens = trained_tokenizer.tokenize(test_domain)
    # ids = trained_tokenizer.convert_tokens_to_ids(tokens)
    # for t, i in zip(tokens, ids):
    #     print(f"{t:10s} -> {i}")

    # input_ids = encoded["input_ids"]
    # tokens = trained_tokenizer.convert_ids_to_tokens(input_ids)
    # print(tokens)

    # # test_domain = "google[.com]"
    # # 예상 결과: ['[CLS]', 'google', '[.com]', '[SEP]']