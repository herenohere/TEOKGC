import torch

from typing import List

from config import args
from triplet import EntityDict
from dict_hub import get_link_graph
from doc import Example


def rerank_by_graph(batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict):

    if args.task == 'wiki5m_ind':
        assert args.neighbor_weight < 1e-6, 'Inductive setting can not use re-rank strategy'

    if args.neighbor_weight < 1e-6:
        return

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]
        n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=1)
        delta = torch.tensor([0.0015 for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)

        n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=2)
        delta = torch.tensor([0.0015 for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)

        n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=3)
        delta = torch.tensor([0.0015 for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)


        if args.task == 'FB15k237':
            n_hop_indices = get_link_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                      entity_dict=entity_dict,
                                                                      n_hop=1)
            n_hop_indices.remove(entity_dict.entity_to_idx(cur_ex.head_id))
            delta = torch.tensor([-0.005 for _ in n_hop_indices]).to(batch_score.device)
            n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

            batch_score[idx].index_add_(0, n_hop_indices, delta)
