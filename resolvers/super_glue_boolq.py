def manual(inputs, include_label=False):
    format = '{}\nquestion: {}? \nanswer:{}'
    label_to_verb = {
        0: ' no',
        1: ' yes',
    }
    result = {
        'prefix': '',
        'concatenator': '\n\n',
        'suffix': '\n\n',
        'resolved_input': [],
        'label': [],
        'idx': [],
        'verbalizers': [' no', ' yes'],
    }
    for input in inputs:
        if include_label:
            resolved_input = format.format(input['passage'], input['question'], label_to_verb[input['label']])
        else:
            resolved_input = format.format(input['passage'], input['question'], '')

        result['resolved_input'].append(resolved_input)
        result['label'].append(input['label'])
        result['idx'].append(input['idx'])

    return result