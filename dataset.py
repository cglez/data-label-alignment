import json
import os
from glob import glob

import fire
from datasets import load_dataset, Value


def download(dataset, split='train', cast=False):
    data_files = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(f'data/datasets/{dataset}/*.csv')}
    ds = load_dataset('csv', data_files=data_files, split=split)
    ds = ds.rename_column('Unnamed: 0', 'id')
    ds = ds.rename_column('text', 'data')

    if cast:
        ds = ds.cast_column("label", Value("string"))
        classes = ds.features["label"]
        ds = ds.map(lambda x: {"label": classes.int2str(x['label'])})

    with open(f"data/{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(ds.to_list(), f, indent=0, ensure_ascii=False)


if __name__ == '__main__':
    fire.Fire(download)
