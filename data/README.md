## Dataset

### Processed data
You can download the processed data `geom_drug.tar.gz` (~2GB) from [here](https://drive.google.com/drive/folders/1WkYIv471SjVwQe6_FfDxFOPC7dSnPY9c?usp=sharing) and unzip them (`tar -zxvf`)
 in the `data` folder as:
``` bash
data
├── geom_drug
│   ├── mol_summary.csv
│   ├── split_by_molid.pt
│   ├── processed.lmdb
│   └── processed_molid2idx.pt
```

### From sdf files
If you want to process the data from the sdf files, you have to further download the `sdf.tar.gz` (~4GB) from [here](https://drive.google.com/drive/folders/1WkYIv471SjVwQe6_FfDxFOPC7dSnPY9c?usp=sharing). Unzip it (~28GB after unzipping) in the `data/geom_drug` folder and remove the `processed.lmdb` and `processed_molid2idx.pt`:
``` bash
data
├── geom_drug
│   ├── mol_summary.csv
│   ├── split_by_molid.pt
│   └── sdf
│       ├── 0.sdf
│       ├── ...
```
Then by running any sampling, training or evaluation script, the data will be processed automatically.