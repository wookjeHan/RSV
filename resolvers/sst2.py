def manual(inputs, include_label=False):
    format = '{} It was {}'
    label_to_verb = {
        0: ' terrible',
        1: ' great',
    }
    result = {
        'prefix': '',
        'concatenator': '\n',
        'suffix': '\n',
        'resolved_input': [],
        'label': [],
        'idx': [],
        'verbalizers': [' terrible', ' great'],
    }
    for input in inputs:
        if include_label:
            resolved_input = format.format(input['sentence'], label_to_verb[input['label']])
        else:
            resolved_input = format.format(input['sentence'], '')

        result['resolved_input'].append(resolved_input)
        result['label'].append(input['label'])
        result['idx'].append(input['idx'])

    return result