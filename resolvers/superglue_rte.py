def manual(inputs, include_label=False):
    format = '{}\nquestion: {}. True or False?\nanswer:{}'
    label_to_verb = {
        0: 'True',
        1: 'False',
    }
    result = {
        'prefix': '',
        'concatenator': '\n\n',
        'suffix': '\n\n',
        'resolved_input': [],
        'label': [],
        'verbalizers': ['True', 'False'],
    }
    for input in inputs:
        if include_label:
            resolved_input = format.format(input['premise'], input['hypothesis'], label_to_verb[input['label']])
        else:
            resolved_input = format.format(input['premise'], input['hypothesis'], '')

        result['resolved_input'].append(resolved_input)
        result['label'].append(input['label'])

    return result