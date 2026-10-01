# Data

The audio is **not** stored in this repository (Zindi's rules do not allow redistribution).

1. Join the challenge and download the files from
   <https://zindi.africa/competitions/swahili-audio-classification/data>:
   `Train.csv`, `Test.csv`, `SampleSubmission.csv`, `Audio.zip`.
2. Put all four files in a Google Drive folder named `swahili_audio` (for Colab),
   or in this `data/` folder (for local runs), and unzip `Audio.zip` there.

`splits.csv` **is** committed: it is the shared stratified train / val / test split created by
`notebooks/01_eda.ipynb`, so every model is trained and evaluated on exactly the same clips.
