from transformers.models.bert.modeling_bert import BertModel
import torch
from torch import nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx) :
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)

        # 초기화
        torch.nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, input) :
        return self.embedding(input)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        P_E = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.arange(0, d_model, step= 2, dtype=torch.float)
        div_term = torch.exp(_2i * (-math.log(10000.0) / d_model)) 

        P_E[:, 0::2] = torch.sin(pos * div_term)
        P_E[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', P_E.unsqueeze(0))
        
    def forward(self, x) :
        seq_len = x.size(1)
        pe_slice = self.pe[:, :seq_len, :]
        return x + pe_slice

class Transformer(nn.Module) :
    def __init__(self, d_model, n_heads, dim_feedforward, num_layers, dropout=0.1) :
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)   

    def forward(self, x, mask=None) :
        if mask is not None and mask.any():
            out = self.encoder(x, src_key_padding_mask=mask)
        else:
            out = self.encoder(x)
        return self.dropout(out)
    
class MTPHead(nn.Module) :
    def __init__(self, d_model, vocab_size, dropout=0.1) :
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) :

        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logtis = self.linear(x)
        return logtis
    
class TPPHead(nn.Module) :
    def __init__(self, d_model, vocab_size, dropout=0.1) :
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x) :

        x = self.dense(x)
        x = self.gelu(x)
        x = self.dropout(x)
        logtis = self.linear(x)
        return logtis
    
class TOVHead(nn.Module):
    def __init__(self, d_model, num_classes=2, dropout=0.1, tov_norm = 'cls'):
        super().__init__()
        self.tov_norm = tov_norm
        if tov_norm == "pool" :
            self.dense = nn.Linear(d_model * 2, d_model * 2)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model * 2, num_classes)
        else :
            self.dense = nn.Linear(d_model, d_model)
            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, sequence_output):

        if self.tov_norm == "pool" :
            max_output = sequence_output.max(dim=1).values
            avg_ouput = sequence_output.mean(dim=1)
            output = torch.cat((max_output, avg_ouput), dim=1)
        else :
            output = sequence_output[:, 0, :]
        
        x = self.dense(output)
        x = self.gelu(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
    
class PretrainedModel(nn.Module) :
    def __init__(self, vocab_size, d_model, n_heads, dim_feedforward, 
                 num_layers, max_len, dropout=0.1, padding_idx=0, tov_norm='cls') :
        super().__init__()

        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.tov_norm = tov_norm

        self.embedding = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = Transformer(d_model, n_heads, dim_feedforward, num_layers, dropout)

        self.mtp_head = MTPHead(d_model, vocab_size)

        self.tpp_head = TPPHead(d_model, vocab_size)

        self.tov_head = TOVHead(d_model, num_classes=2, dropout=dropout, tov_norm=self.tov_norm)

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx)
    
    def forward(self, input_ids, task_type='ALL'):
        token_embed = self.embedding(input_ids)
        x = self.positional_encoding(token_embed)
        padding_mask = self.create_padding_mask(input_ids)
        encoder_output = self.transformer(x, mask=padding_mask)

        outputs = {}

        # Task 1: MTP
        if task_type == 'MTP' or task_type == 'ALL':
            outputs['mtp_logits'] = self.mtp_head(encoder_output)
            
        # Task 2: TTP
        if task_type == 'TPP' or task_type == 'ALL':
            outputs['ttp_logits'] = self.tpp_head(encoder_output)
            
        # Task 3: TOV
        if task_type == 'TOV' or task_type == 'ALL':
            outputs['tov_logits'] = self.tov_head(encoder_output)

        # 단일 태스크를 요구했을 경우 dict 대신 로짓 텐서 자체를 반환
        if len(outputs) == 1 and task_type != 'ALL':
            return list(outputs.values())[0]
            
        return outputs


class FinetuningHead(nn.Module) :
    def __init__(self, input_dim, d_model, dropout) :
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.dense1 = nn.Linear(input_dim, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model * 2, 2)

    def forward(self, encoder_output) :

        x = self.dense1(encoder_output)
        x = torch.relu(x)
        x = self.dropout(x)

        logits = self.classifier(x)
        return logits
    
class FineTuningModel(nn.Module):
    def __init__(self, pretrain_model_t=None, pretrain_model_c=None, use_bert=False,
                 dropout=0.1, padding_idx=0, clf_norm = 'cls', freeze_backbone=False):
        super().__init__()

        self.padding_idx = padding_idx
        self.clf_norm = clf_norm
        self.use_token = pretrain_model_t is not None
        self.use_char = pretrain_model_c is not None
        self.use_bert = use_bert

        sample_model = pretrain_model_t if self.use_token else pretrain_model_c # Extract d_model value
        d_model = sample_model.d_model

        num_active_paths = sum([self.use_token, self.use_char])
        dim_per_path = d_model * 2 if self.clf_norm == 'pool' else d_model
        total_input_dim = dim_per_path * num_active_paths

        if self.use_bert :
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self._set_grad(self.bert, False) # freeze bert
            total_input_dim += self.bert.config.hidden_size

        # --- Token Path Components ---
        # load_pretrain(pretrain_model_t)
        if self.use_token:
            self.transformer_encoder_t = pretrain_model_t.transformer
            self.embedding_t = pretrain_model_t.embedding
            self.positional_encoding_t = pretrain_model_t.positional_encoding
            if freeze_backbone:
                self._set_grad(self.transformer_encoder_t, False)
                self._set_grad(self.embedding_t, False)
                self._set_grad(self.positional_encoding_t, False)

        # --- Character Path Components ---
        # load_pretrain(pretrain_model_c)
        if self.use_char:
            self.transformer_encoder_c = pretrain_model_c.transformer
            self.embedding_c = pretrain_model_c.embedding
            self.positional_encoding_c = pretrain_model_c.positional_encoding
            if freeze_backbone:
                self._set_grad(self.transformer_encoder_c, False)
                self._set_grad(self.embedding_c, False)
                self._set_grad(self.positional_encoding_c, False)
        
        # DGA 분류 헤드 연결
        self.classifier_head = FinetuningHead(
            input_dim=total_input_dim,
            d_model=d_model,
            dropout=dropout
        )

    def _set_grad(self, module, requires_grad=False):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def create_padding_mask(self, input_ids):
        return (input_ids == self.padding_idx).to(input_ids.device)

    def forward(self, input_ids_t=None, input_ids_c=None, 
                bert_input_ids=None, bert_mask=None):
        features = []

        # --- 1. Token Path (X_t) 처리 ---
        if self.use_token and input_ids_t is not None:
            t_embed = self.embedding_t(input_ids_t)
            t_x = self.positional_encoding_t(t_embed)
            t_mask = self.create_padding_mask(input_ids_t)
            t_out = self.transformer_encoder_t(t_x, mask=t_mask)

            if self.clf_norm == 'pool':
                # Max pool + Mean pool (d_model * 2)
                t_feat = torch.cat([t_out.max(dim=1).values, t_out.mean(dim=1)], dim=1)
            else:
                # CLS Token (d_model * 1)
                t_feat = t_out[:, 0, :]
            features.append(t_feat)

        # --- 2. Character Path (X_c) 처리 ---
        if self.use_char and input_ids_c is not None:
            c_embed = self.embedding_c(input_ids_c)
            c_x = self.positional_encoding_c(c_embed)
            c_mask = self.create_padding_mask(input_ids_c)
            c_out = self.transformer_encoder_c(c_x, mask=c_mask)

            # print(f"c_x shape: {c_x.shape}") # 예상: [128, 77, 256]
            # print(f"c_mask shape: {c_mask.shape}") # 예상: [128, 77]

            if self.clf_norm == 'pool':
                # Max pool + Mean pool (d_model * 2)
                c_feat = torch.cat([c_out.max(dim=1).values, c_out.mean(dim=1)], dim=1)
            else:
                # CLS Token (d_model * 1)
                c_feat = c_out[:, 0, :]
            features.append(c_feat)

        if self.use_bert and bert_input_ids is not None :
            with torch.no_grad():
                bert_out = self.bert(bert_input_ids, attention_mask=bert_mask)
                bert_feat = bert_out.last_hidden_state[:, 0, :]
            features.append(bert_feat)

        combined_output = torch.cat(features, dim=1) if len(features) > 1 else features[0]

        # # 디버깅용 출력 (한 번만 확인하고 지우셔도 됩니다)
        # if not hasattr(self, '_size_checked'):
        #     print(f"DEBUG: Combined output shape: {combined_output.shape}")
        #     print(f"DEBUG: Head input_dim: {self.classifier_head.dense1.in_features}")
        #     self._size_checked = True

        return self.classifier_head(combined_output)
    
    def set_backbone_freezing(self, freeze=True):
        """Backbone의 학습 여부를 외부에서 조절하는 함수"""
        trainable = not freeze
        
        if self.use_token:
            for p in self.transformer_encoder_t.parameters(): p.requires_grad = trainable
            for p in self.embedding_t.parameters(): p.requires_grad = trainable
            for p in self.positional_encoding_t.parameters(): p.requires_grad = trainable

        if self.use_char:
            for p in self.transformer_encoder_c.parameters(): p.requires_grad = trainable
            for p in self.embedding_c.parameters(): p.requires_grad = trainable
            for p in self.positional_encoding_c.parameters(): p.requires_grad = trainable
            
        status = "고정(Frozen)" if freeze else "해제(Unfrozen)"
        print(f"--- Backbone이 {status} 상태로 변경되었습니다. ---")