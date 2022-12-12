class Truncator:
    def __init__(self, tokenizer, max_seq_len, max_sample_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_sample_seq_len = max_sample_seq_len

    def _tokenize(self, input):
        if isinstance(input, list):
            tok_result = self.tokenizer(input, padding=True)
            input_ids = tok_result['input_ids']
            max_tok_len = len(input_ids[0])

            return input_ids, max_tok_len
        else:
            tok_result = self.tokenizer(input)
            input_ids = tok_result['input_ids']
            tok_len = len(input_ids)

            return input_ids, tok_len

    def _composite(self, resolved_shots, input):
        verbalizers = resolved_shots['verbalizers']
        resolved_inputs = resolved_shots['resolved_input']
        labels = resolved_shots['label']
        shot_num = len(labels)

        demonstrations = [resolved_inputs[i] + verbalizers[labels[i]] for i in range(shot_num)]
        composited_input = resolved_shots['concatenator'].join(demonstrations)

        return resolved_shots['prefix'] + composited_input + resolved_shots['suffix'] + input

    def truncate(self, resolved_shots_batch, input_batch):
        truncated_batch = []
        verbalizers = input_batch['verbalizers']

        for resolved_shots, input in zip(resolved_shots_batch, input_batch['resolved_input']):
            composited_input = self._composite(resolved_shots, input)
            composited_inputs = [composited_input + verbalizer for verbalizer in verbalizers]
            _, max_tok_len = self._tokenize(composited_inputs)

            if max_tok_len <= self.max_seq_len:
                truncated_batch.append(composited_input)
            else:
                # 1) truncate each shots indivisually
                resolved_inputs = resolved_shots['resolved_input']
                word_labels = [verbalizers[int_label] for int_label in resolved_shots['label']]

                for idx, (resolved_input, label) in enumerate(zip(resolved_inputs, word_labels)):
                    _, tok_len = self._tokenize(resolved_input + label)
                    _, label_tok_len = self._tokenize(label)

                    if self.max_sample_seq_len < tok_len:
                        input_ids, tok_len = self._tokenize(resolved_input)
                        input_ids = input_ids[:self.max_sample_seq_len - label_tok_len - 2] # 2 is a margin
                        resolved_shots['resolved_input'][idx] = self.tokenizer.decode(input_ids)
                
                # 2) truncate the input
                _, tok_len = self._tokenize([input + verbalizer for verbalizer in verbalizers])
                _, verb_tok_len = self._tokenize(verbalizers)

                if self.max_sample_seq_len < tok_len:
                    input_ids, tok_len = self._tokenize(input)
                    input_ids = input_ids[:self.max_sample_seq_len - verb_tok_len - 2] # 2 is a margin
                    truncated_input = self.tokenizer.decode(input_ids)
                else:
                    truncated_input = input

                # 3)
                composited_input = self._composite(resolved_shots, truncated_input)
                composited_inputs = [composited_input + verbalizer for verbalizer in verbalizers]

                # 4)
                _, max_tok_len = self._tokenize(composited_inputs)

                if self.max_seq_len < max_tok_len:
                    input_ids, tok_len = self._tokenize(composited_input)
                    truncated_composited_input = self.tokenizer.decode(input_ids[tok_len - self.max_seq_len + verb_tok_len + 2:]) # 2 is a margin
                    truncated_batch.append(truncated_composited_input)
                else:
                    truncated_batch.append(composited_input)

        return truncated_batch