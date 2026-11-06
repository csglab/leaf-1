"""MLM data collator used by LEAF-1 pretraining.

Base tokens are masked at `mlm_probability` (paper: 0.15); both [CUT] tokens of a
fragment are masked jointly at `cut_token_probability` (paper: 0.30) so the model
cannot infer one boundary trivially from the other. The modality token (column 1)
is never masked.
"""

import torch


class DataCollatorForDNAFragmentModelingNoDecay:
    def __init__(self, tokenizer, mlm_probability=0.15, cut_token_probability=0.30):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.cut_token_probability = cut_token_probability
        self.cut_token_id = tokenizer.convert_tokens_to_ids('[CUT]')

    def mask_tokens(self, inputs, labels):
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        cut_bool_matrix = (inputs == self.cut_token_id)
        n_cut_per_row = cut_bool_matrix.sum(dim=1)
        has_cut = bool((n_cut_per_row > 0).any())
        # two [CUT] tokens per row, or none (nocut)
        if has_cut and not bool((n_cut_per_row == 2).all()):
            raise ValueError(f"expected 2 [CUT] tokens per row (or none); got {n_cut_per_row.tolist()}")

        if has_cut:
            cut_positions = cut_bool_matrix.nonzero(as_tuple=True)
            batch_indices, cut_indices = cut_positions

            probability_matrix[batch_indices, cut_indices] = self.cut_token_probability

            # Joint masking: each row either masks both [CUT] tokens or neither.
            # `cut_indices` arrives as a flat tensor of length 2 * batch_size (two
            # boundaries per row). Reshape to (batch_size, 2) so indexing into
            # probability_matrix touches both CUT positions of each row together.
            cut_indices = cut_indices.view(-1, 2)
            for i in range(len(inputs)):
                mask_both = torch.bernoulli(torch.tensor(self.cut_token_probability)).bool()
                probability_matrix[i, cut_indices[i]] = 1.0 if mask_both else 0.0

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Never mask the modality token (column 1).
        probability_matrix[:, 1] = 0.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def __call__(self, batch):
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        inputs = torch.stack([item['input_ids'] for item in batch])
        labels = inputs.clone()
        inputs, labels = self.mask_tokens(inputs, labels)
        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}
