# PV Power Forecasting

## Key Concepts

:star2: **Large PV data**: We use a large-scale PV dataset. The dataset contains 1-hour resolution data mainly from [DKASC](https://dkasolarcentre.com.au/) (Desert Knowledge Australia Solar Centre)

:star2: **Validated on the Various Sites**: The trained model is validated on 4 different countries (England, Germany, Korea, and USA) and each countries has multiple sites. 

![Algorithm Overview](pic/fig1.png)

*Overview of our transfer learning approach for PV power forecasting*

## Results

(add contents after conducting experiments...)

## Getting Started

Our model basically comes from the PatchTST model. We modified the model to fit the PV power forecasting task.
Though you can follow the instructions from the original PatchTST repo, we provide the detailed instructions for the PV power forecasting task.

1. Install pytorch. You can install pytorch from the official website: https://pytorch.org/get-started/locally/  
   (However, we tested on pytorch 2.0.1, CUDA 11.7.)

2. Install requirements. ```pip install -r requirements.txt```

3. Download data (Please tell us if you need the data.).
4. Put the data in the ```./data``` directory.  
   The directory structure should be as follows:
   ```
    ./data
    ├── DKASC_AliceSprings
    │   ├── CSV file 01
    │   ├── CSV file 02
    │   └── ...
    ├── GIST_dataset
    │   ├── CSV file 01
    │   ├── CSV file 02
    │   └── ...
   
   ```
   The detailed information about the data preprocessing is in the `./data_preprocessing/DATA_PREPROCESSING.md` .

5. Run `./run_longExp.py` for a single experiment.   
   You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows, prediction lengths etc.).

6. (Optional) If you want to run experiments on sequentially, you can use the bash files on `./scripts/PatchTST` directory.  
   For example, if you want to run a single site of DKASC experiment (`79-Site_DKA-M6_A-Phase.csv`) with changing the prediction length, you can run the following command:  
   ```
   sh ./scripts/PatchTST/DKASC_single.sh
   ``` 


## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST

https://dkasolarcentre.com.au/

## Contact

If you have any questions or concerns, please contact us: bakseongho@gm.gist.ac.kr or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@inproceedings{
adding soon
}
```
