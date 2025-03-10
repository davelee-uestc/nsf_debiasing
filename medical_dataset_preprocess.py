import os
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def generate_metadata_chexpert(data_path, test_pct=0.15, val_pct=0.1):
    logging.info("Generating metadata for CheXpert No Finding prediction...")
    chexpert_dir = Path(os.path.join(data_path, "chexpert"))
    assert (chexpert_dir/'train.csv').is_file()
    assert (chexpert_dir/'train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'valid/patient64636/study1/view1_frontal.jpg').is_file()
    assert (chexpert_dir/'CHEXPERT DEMO.xlsx').is_file()

    df = pd.concat([pd.read_csv(chexpert_dir/'train.csv'), pd.read_csv(chexpert_dir/'valid.csv')], ignore_index=True)

    df['img_filename'] = df['Path'].astype(str).apply(lambda x:  x[x.index('/')+1:])
    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:])).astype(str)
    df = df[df.Sex.isin(['Male', 'Female'])]
    details = pd.read_excel(chexpert_dir/'CHEXPERT DEMO.xlsx', engine='openpyxl')[['PATIENT', 'PRIMARY_RACE']]
    details['subject_id'] = details['PATIENT'].apply(lambda x: x[7:]).astype(int).astype(str)

    df = pd.merge(df, details, on='subject_id', how='inner').reset_index(drop=True)

    def cat_race(r):
        if isinstance(r, str):
            if r.startswith('White'):
                return 0
            elif r.startswith('Black'):
                return 1
        return 2

    df['ethnicity'] = df['PRIMARY_RACE'].apply(cat_race)
    attr_mapping = {'Male_0': 0, 'Female_0': 1, 'Male_1': 2, 'Female_1': 3, 'Male_2': 4, 'Female_2': 5}
    df['spurious'] = (df['Sex'] + '_' + df['ethnicity'].astype(str)).map(attr_mapping)
    df['y'] = df['No Finding'].fillna(0.0).astype(int)

    train_val_idx, test_idx = train_test_split(df.index, test_size=test_pct, random_state=42, stratify=df['spurious'])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_pct/(1-test_pct), random_state=42, stratify=df.loc[train_val_idx, 'spurious'])

    df['split'] = 0
    df.loc[val_idx, 'split'] = 1
    df.loc[test_idx, 'split'] = 2

    df.to_csv(os.path.join(chexpert_dir, "metadata.csv"), index=False)

if __name__=='__main__':
    generate_metadata_chexpert('path/to/chexpert')