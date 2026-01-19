<div align="center">
  <h2><b> Code for Paper:</b></h2>
  <h2><b> Signals Meet Sentences: Medical Time Series Classification with Integrated LLM Priors </b></h2>
</div>

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download *APAVA, ADFTD, PTB, and PTB-XL* from [**Medformer**](https://github.com/DL4mHealth/Medformer). **All the datasets are well pre-processed**. For the *TDBrain and MIMIC* dataset, we refer to first downloading the raw dataset from [**TDBrain**](https://brainclinics.com/resources/) and [**MIMIC**](https://physionet.org/content/mimic-iv-ecg/1.0/), respectively. Then, preprocess them using ```data_preprocessing/TDBRAIN_preprocessing.ipynb``` and ```data_preprocessing/MIMIC-IV_preprocessing.ipynb```. Finally, place all datasets under the folder ```./dataset```. Here, we provide an example, i.e., **APAVA** dataset.
3. Get the VLM embedding. You can use comments below './scripts/get_vlm_emb' to get the LLM embeddings of all datasets. Such as ```bash scripts/get_vlm_emb/APAVA.sh```
4. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. Such as ```bash ./scripts/APAVA.sh ``` to get the result of  **APAVA**. You can find the training history and results under the './logs' folder.

## Acknowledgement

This project is based on the code in the repository [**Medformer**](https://github.com/DL4mHealth/Medformer).
Thanks a lot for their amazing work!

***Please also star their project and cite their paper if you find this repo useful.***
```
@article{wang2024medformer,
  title={Medformer: A multi-granularity patching transformer for medical time-series classification},
  author={Wang, Yihe and Huang, Nan and Li, Taida and Yan, Yujun and Zhang, Xiang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={36314--36341},
  year={2024}
}
```

