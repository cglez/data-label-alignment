import json
import os
from glob import glob

import fire
from datasets import load_dataset, Value


def to_json(dataset, split='train', label='label'):
    data_files = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(f'data/datasets/{dataset}/*.csv')}
    ds = load_dataset('csv', data_files=data_files, split=split)

    if 'id' not in ds.column_names:
        ds = ds.rename_column('Unnamed: 0', 'id')
    if label != 'label':
        ds = ds.rename_column(label, 'label')
    ds = ds.rename_column('text', 'data')
    ds = ds.select_columns(['id', 'data', 'label'])

    if ds.features['label'] == Value('int64'):
        ds = ds.cast_column('label', Value('string'))

    with open(f'data/{dataset}.json', 'w', encoding='utf-8') as f:
        json.dump(ds.to_list(), f, indent=0, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire(to_json)
