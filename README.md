# Boost Camp 2기 31조 Level 1 - Mask Image Classification Competition

### Installation

---

All requirements should be detailed in requirements.txt

```
pip install -r requirements.txt
```

### Archive contents

---

#### Architecture1 : swin_base_patch4_window12_384

- module : contains original code, trained models etc

```
module
|-- data_utils
|   |-- data_loaders.py
|   |-- datasets.py
|   `-- make_df_sep_val_trn.py
|-- model
|   |-- loss.py
|   `-- models.py
|-- inference_test.py
|-- train.py
`-- trainer
    |-- __init__.py
    |-- custom_scheduler.py
    `-- trainer.py
```

- `data_utils/` : data_loader, dataset 등 data를 불러오는데 필요한 파일들이 있는 폴더
- `model/` : model 선언과 loss class들이 있는 폴더
- `inference_test.py` : trained된 model들을 바탕으로 submission.csv 파일 생성
- `train.py` : 설정한 모델을 바탕으로 학습된 모델 파라미터를 저장
- `trainer/` : `train.py` 에서 학습에 필요한 함수들 모음
