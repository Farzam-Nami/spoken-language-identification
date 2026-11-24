# Dataset setup

The input corpus consists of MP3 audiobook excerpts in four languages. The raw corpus is available through the [dataset link supplied by the project author](https://drive.google.com/drive/folders/1Jb0PLfvtUfYd05odIsg2IR3q8nEmPfzx?usp=sharing). Access and redistribution permissions have not been independently verified. The raw audio is not bundled. The supplied `features_table.csv` is included in the local project archive and remains ignored by Git pending permission and review of file identifiers.

Expected layout (case-sensitive on Linux):

```text
datasets/
└── Dataset/
    ├── German/
    │   ├── Male/       # *.mp3
    │   └── Female/     # *.mp3
    ├── Italian/
    │   ├── Male/
    │   └── Female/
    ├── Korean/
    │   ├── Male/
    │   └── Female/
    └── Spanish/
        ├── Male/
        └── Female/
```

Run `notebooks/Data_Cleaning_and_Feature_Extraction.ipynb` from the `notebooks/` working directory. It generates `ProcessedDataset/`, `metadata_manifest.csv`, `metadata_clean_manifest.csv` and `features_table.csv` here; `feature_extraction_failed.csv` is generated only if failures occur. The processed paths in a newly generated feature table are relative to `notebooks/`. Existing processed WAV files are reused.

The supplied feature table is already at `datasets/features_table.csv` in the local archive. If using a Git clone, place a permitted copy at that path. It must contain `language`, `gender`, `processed_path` and the same 128 numeric features. Classification and clustering use the numeric features only and do not open the referenced WAV files. The original recorded table has 655 rows and 131 columns. Raw and derived data files are ignored by Git by default. The included aggregate results can be viewed without any input data.

## Feature table validation

The supplied CSV contains 655 rows, 131 columns (128 numeric features and 3 metadata fields), 655 distinct processed paths, no duplicate rows, no missing values and no non-finite numeric values. Class counts are German 158, Italian 159, Korean 170 and Spanish 168. The existing `processed_path` values are retained verbatim; classification and clustering exclude these identifiers from model inputs. To regenerate features from raw audio, run the first notebook with the original corpus available.
