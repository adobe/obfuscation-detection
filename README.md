# Obfuscation Detection with Deep Learning

Command obfuscation is a technique to make a piece of code intentionally hard-to-read, but still 
execute the same functionality. Malicious attackers often abuse obfuscation to make their malicious 
software (malware) evasive to traditional malware detection techniques. This creates a headache for 
defenders since attackers can create a virtually infinite number of ways to obfuscate their malware. 
Traditional malware detection techniques are often rule-based, rendering them inflexible to new 
types of malware and obfuscation techniques. Deep learning has been used in various domains to 
create models that are dynamic and can adapt to new types of information. Our project uses deep 
learning techniques to detect command obfuscation.

## Installation

### Pre-requisites
1. Make empty directories for runtime
```
mkdir data
mkdir data/prep
mkdir data/processed_tensors
mkdir data/scripts
mkdir models
```

2. Download the Powershell Corpus: https://aka.ms/PowerShellCorpus. Unzip the file into the `data` directory

3. Download PS corpus labels: https://github.com/danielbohannon/Revoke-Obfuscation. Clone the repo in the same-level directory as this repo. The labels are found in the `DataScience` folder.

4. Download DOSfuscated commands: https://github.com/danielbohannon/Invoke-DOSfuscation/tree/master/Samples. Download the four `STATIC_#_of_4_*` files and put them inside the `data` directory.

5. We also use internal Adobe command line executions as a bulk of our training data. We may not release this dataset to the public, so we encourage you to either not exclude this dataset or find an open-source command prompt dataset on the internet.

### Data Prep

`cd data-prep`, then run the following python scripts in order:

1. `python char_frequency.py`: creates a .txt file of each charcter's frequency in the dataset

2. `python char_dict.py`: creates a python dict of the most common characters mapped to a numeric index

3. `python ps_data_preprocess.py`: creates tensors for the powershell corpus dataset

4. `python dos_data_preprocess.py`: creates tensors for the DOSfuscated commands

5. `python hubble_data_preprocess.py` and `python cb_data_preprocess.py`: creates tensors for internal Adobe data. Replace this step with other data you may find on the internet.

6. `python data_preprocess.py`: creates train/dev/test tensor split by accumulating all tensors.


### Running

`models.py`: contains the different model architectures we experimented with

`main.py`:
- `--model` - choose a model architecture
- `--model-file` - filename of model checkpoint
- `--cuda-device` - which cuda device to use
- Choose one of:
    - `--reset` - start training from scratch
    - `--eval` - evaluate on best checkpoint on train/dev
    - `--analyze` - create fp/fn files from dev
    - `--run` - run on real scripts on `test-scripts` dir
    - `--test` - run model on test set
    - none - continue training model from checkpoint
