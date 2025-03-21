from abc import ABC
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
from triplet_mask import construct_mask
def build_model(args) -> nn.Module:
    return CustomBertModel(args)
class ModelOutput:
    def __init__(self, logits=None, labels=None, inv_t=None, hr_vector=None, tail_vector=None):
        self.logits = logits
        self.labels = labels
        self.inv_t = inv_t
        self.hr_vector = hr_vector
        self.tail_vector = tail_vector
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
class AdaptiveFusion(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFusion, self).__init__()
        self.gate = nn.Linear(input_dim * 2, input_dim)
    def forward(self, encoder_features, attention_features):
        attention_features = attention_features.squeeze(1)
        combined = torch.cat((encoder_features, attention_features), dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined))
        fused_features = gate_weights * encoder_features + (1 - gate_weights) * attention_features
        return fused_features
class DeformableSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_deformable_points):
        super(DeformableSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_deformable_points = num_deformable_points
        self.offset_fc = nn.Linear(embed_dim, num_heads * num_deformable_points * 2)
        self.attention_fc = nn.Linear(embed_dim, num_heads * num_deformable_points)
        self.softmax = nn.Softmax(dim=-1)
        self.output_fc = nn.Linear(embed_dim, embed_dim)
    def deformable_sample_and_attend(self, x, offset, attention_weights):
        B, N, num_heads, num_points, _ = offset.shape
        _, _, C = x.shape
        output = torch.zeros_like(x)
        for i in range(num_heads):
            sampling_locations = offset[:, :, i, :, :]
            weights = attention_weights[:, :, i, :]
            sampled_features = self.bilinear_sample(x, sampling_locations)
            weighted_sum = (sampled_features * weights.unsqueeze(-1)).sum(dim=-2)
            output += weighted_sum
        return output

    def bilinear_sample(self, x, sampling_locations):
        B, N, num_points, _ = sampling_locations.shape
        _, _, C = x.shape

        sampled_features = torch.zeros(B, N, num_points, C, device=x.device)

        for i in range(B):
            for j in range(N):
                locs = sampling_locations[i, j]
                for k in range(num_points):
                    x_idx, y_idx = locs[k].long()
                    sampled_features[i, j, k] = x[i, 0, x_idx]
        return sampled_features

    def forward(self, x):
        if x.dim() == 2:
            B, C = x.shape
            N = 1
            x = x.unsqueeze(1)
        else:
            B, N, C = x.shape
        offset = self.offset_fc(x).view(B, N, self.num_heads, self.num_deformable_points, 2)
        attention_weights = self.attention_fc(x).view(B, N, self.num_heads, self.num_deformable_points)
        attention_weights = self.softmax(attention_weights)
        output = self.deformable_sample_and_attend(x, offset, attention_weights)
        output = self.output_fc(output)
        return output
class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)

        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        self.deformable_attention = DeformableSelfAttention(embed_dim=self.config.hidden_size,
                                                            num_heads=args.head_num,
                                                            num_deformable_points=args.deformable_points)
        self.adaptive_fusion = AdaptiveFusion(input_dim=self.config.hidden_size)


    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        attention_output = self.deformable_attention(cls_output)

        #attention_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

        fused_output = self.adaptive_fusion(cls_output, attention_output)

        return fused_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(args.p_weight).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector