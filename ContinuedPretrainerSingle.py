import torch
from transformers.trainer import Trainer
from packaging import version
from transformers import DefaultDataCollator, DataCollatorForLanguageModeling
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler
from transformers.utils import logging
from torch import nn
from typing import Dict, Union, Any
import torch.nn.functional as F
from transformers.file_utils import is_apex_available


if is_apex_available():
    from apex import amp


logger = logging.get_logger(__name__)


class TCMContinuedPreTrainer(Trainer):
    def __init__(self,
                train_cls: bool = False,
                contrast_layer: int = 8,
                temperature: float = 1.0,
                use_transliteration_emb: bool = False,
                tcm_loss_weight = 1.0,
                use_contrastive = True,
                use_lm = True,
                **kwargs):
        logger.debug("Initialising trainer")
        super().__init__(**kwargs)

        self.temperature = temperature
        self.train_cls = train_cls
        self.contrast_layer = contrast_layer
        self.use_transliteration_emb = use_transliteration_emb
        self.tcm_loss_weight = tcm_loss_weight
        self.use_contrastive = use_contrastive
        self.use_lm = use_lm


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()

        loss = torch.zeros([]).to(self.args.device)

        if not self.use_transliteration_emb:
            text_inputs = {'input_ids': inputs.pop('input_ids_1'),
                  'attention_mask': inputs.pop('attention_mask_1'),
                  'labels': inputs.pop('labels_1'),
                  'token_type_ids': inputs.pop('token_type_ids_1'),
                  'output_hidden_states': True,
                  }
            trans_inputs = {'input_ids': inputs.pop('input_ids_2'),
                  'attention_mask': inputs.pop('attention_mask_2'),
                  'labels': inputs.pop('labels_2'),
                  'token_type_ids': inputs.pop('token_type_ids_2'),
                  'output_hidden_states': True,
                  }
        else:
            text_inputs = {'input_ids': inputs.pop('input_ids_1'),
                  'attention_mask': inputs.pop('attention_mask_1'),
                  'labels': inputs.pop('labels_1'),
                  'token_type_ids': inputs.pop('token_type_ids_1'),
                  'output_hidden_states': True,
                  'is_for_transliteration_input': False
                  }
            trans_inputs = {'input_ids': inputs.pop('input_ids_2'),
                  'attention_mask': inputs.pop('attention_mask_2'),
                  'labels': inputs.pop('labels_2'),
                  'token_type_ids': inputs.pop('token_type_ids_2'),
                  'output_hidden_states': True,
                  'is_for_transliteration_input': True
                  }

        pool_mask_1 = inputs.pop('pool_mask_1')
        pool_mask_2 = inputs.pop('pool_mask_2')

        text_inputs = self._prepare_inputs(text_inputs)
        trans_inputs = self._prepare_inputs(trans_inputs)

        results1 = model(**text_inputs)
        results2 = model(**trans_inputs)

        # print(results1)
        # print()
        with self.compute_loss_context_manager():
            # doing mlm
            if self.use_lm:
                lm_loss = results1['loss'] + results2['loss']
                loss = loss + lm_loss

            # print("lm_loss_1: ", results1['loss'])
            # print("lm_loss_2: ", results2['loss'])

            # doing contrastive
            if self.use_contrastive:
                tcm_loss = self.do_tcm_forward(results1, results2, pool_mask_1, pool_mask_2)
                # print("tcm_loss: ", tcm_loss)
                loss = loss + self.tcm_loss_weight * tcm_loss

            # to avoid error because not all parameters contribute to the loss
            if not self.use_lm:
                for p in model.parameters():
                    loss += 0.0 * p.sum()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    # doing forward for TCM loss
    def do_tcm_forward(self, results1, results2, pool_mask_1, pool_mask_2):
        outputs1 = results1['hidden_states'][self.contrast_layer]
        outputs2 = results2['hidden_states'][self.contrast_layer]
        if self.train_cls:
            outputs1 = outputs1[:, 0, :]
            outputs2 = outputs2[:, 0, :]
        else:
            outputs1 = _mean_pool(outputs1, pool_mask_1)
            outputs2 = _mean_pool(outputs2, pool_mask_2)
            
        tcm_loss = seq_contrast(outputs1, outputs2, self.temperature)

        return tcm_loss


# from https://github.com/microsoft/COCO-LM/issues/2
def get_seq_label(sim_matrix):
    bsz = sim_matrix.size(0)
    seq_label = torch.arange(0, bsz, device=sim_matrix.device).view(-1, 2)
    seq_label[:, 0] = seq_label[:, 0] + 1
    seq_label[:, 1] = seq_label[:, 1] - 1
    # label is [1, 0, 3, 2, 5, 4, ...]
    seq_label = seq_label.view(-1)
    return seq_label

# from https://github.com/microsoft/COCO-LM/issues/2
def seq_contrast(out_1, out_2, temperature):
    batch_size = out_1.size(0)
    # [2*B, D], orig and span interleavely
    global_out = torch.cat([out_1, out_2], dim=-1).view(2 * batch_size, -1)
    # [2*B, 2*B]
    sim_matrix = torch.mm(global_out, global_out.t()) / temperature
    global_batch_size = sim_matrix.size(0)
    sim_matrix.masked_fill_(torch.eye(global_batch_size, device=sim_matrix.device, dtype=torch.bool), float('-inf'))
    truth = get_seq_label(sim_matrix)
    truth.requires_grad = False

    # Using torch.log_softmax and torch.nn.NLLLoss
    log_softmax_sim_matrix = torch.log_softmax(sim_matrix, dim=-1, dtype=torch.float32)
    nll_loss = torch.nn.NLLLoss(reduction='mean')
    contrast_loss = nll_loss(log_softmax_sim_matrix, truth) * 0.5

    return contrast_loss

# 1 is the sequence token and 0 is the special token
def _mean_pool(data, mask):
    mask = mask.to(data.device)
    mask.requires_grad = False
    return (data * mask.unsqueeze(2).float()).sum(dim=1) / mask.sum(dim=1).view(-1, 1)
