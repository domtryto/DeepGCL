# DeepGCL
DeepGCL
we propose a novel multi-view graph contrastive learning framework for DTA prediction. 

# Raw Data
We use two main datasets:PDBBind and BDB2020+
From the PDBBind dataset,PDBBind2020 is split into train set and valid set
For evaluation, we use three independent test sets, PDBBind2016, PDBBind2013 and BDB2020+
Download Link:  https://pan.baidu.com/s/1xO-HGLDHb0VgmDr70vSXsg?pwd=5210 

# Data Preprocessing
We have already preprocessed all datasets and provide them in '.pt' format for direct use.
Our data is uploaded to Google Drive. https://drive.google.com/drive/folders/161xCpbKUdkkxY-j_nGC9KfA2vPfaIPZd?usp=drive_link
The path should be saved in the data/processed folder under the root directory. 

If you prefer to preprocess the raw data yourself, you need to manually modify the file paths and output filenames in create_data.py according to your data location.
```python /data/cerate_data.py ```
This will generate all '.pt' files in the data/processed/ directory.


# Train
If you want to train the model, run python main5.py. Our training set at https://drive.google.com/drive/folders/161xCpbKUdkkxY-j_nGC9KfA2vPfaIPZd?usp=drive_link

The model part is located in MMPD_DTA5.py 
```python main5.py```

Our test implementation is in test.py, and the model data is stored on Google Drive. https://drive.google.com/drive/folders/1Emwvchxjqhak6xHQUMP8gHFntns9VPkD?usp=drive_link

The path should be saved in the result/best folder under the root directory. 

# Evaluation

We provide pre-trained model weights for direct evaluation on all test sets.
To evaluate the model on all test sets, simply run:
```python test.py```

