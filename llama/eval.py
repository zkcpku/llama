# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMAEval:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def eval(
        self,
        inputs,
        max_gen_len: int,
        labels = None
    ) -> Dict[str, float]:
        """
        Evaluate the model using the ppl and next token accuracy metrics on the given inputs and labels.
        if labels is None, then the model will be evaluated on the inputs themselves.
        """
        bsz = len(inputs)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # add bos and eos to allign
        input_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in inputs]
        
        min_input_size = min([len(t) for t in input_tokens])
        max_input_size = max([len(t) for t in input_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_input_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(input_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id

        if labels is None:
            label_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in labels]
            label_tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
            for k, t in enumerate(label_tokens):
                label_tokens[k, : len(t)] = torch.tensor(t).long()
            label_text_mask = label_tokens != self.tokenizer.pad_id
        else:
            # label_tokens is shifted by one from tokens
            label_tokens = tokens[:, 1:]
            label_text_mask = input_text_mask[:, 1:]
        
        logits = self.model.forward_logits(tokens, 0)
        # TODO: check dim of logits, suppose [bsz, seq_len, vocab_size]
        logits = logits[:, :-1, :]
        logits = logits.reshape(-1, logits.shape[-1])
        label_tokens = label_tokens.reshape(-1)
        label_text_mask = label_text_mask.reshape(-1)
        # ppl
        ppl = torch.nn.functional.cross_entropy(logits, label_tokens, reduction='none')
        # next token accuracy
        next_token_acc = torch.argmax(logits, dim=-1) == label_tokens
        next_token_acc = next_token_acc.float()

        ppl = ppl * label_text_mask.float()
        next_token_acc = next_token_acc * label_text_mask.float()

        token_length = label_text_mask.sum().item()

        return {
            'ppl': ppl.sum().item() / token_length,
            'next_token_acc': next_token_acc.sum().item() / token_length,
            'token_length': token_length
        }






    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
