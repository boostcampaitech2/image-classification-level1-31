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

##### 리더보드에 제출할 모델을 재현할 방법

```
1. tools의 face_crop.py , face_crop_eval.py 실행
2. make_df_sep_val_trn.py 실행으로 train / test 용 csv split
3. train.py 실행
4. 트레이닝된 model을 모아 inferece_test.py를 통해 앙상블 후 submission파일로 저장
```

#### Architecture2 : tf_efficientnet_b4_ns

```
modules
|-- datset.py
|-- loss.py
|-- model.py
|-- train.py
|-- inference.py
|-- model
|   |-- exp
|       |-- best.pth
|       |-- config.json
|       |-- last.pth
|   |-- exp1
|-- output
|   |-- output.csv
```

- `baseline/` : 전체 코드가 들어있는 폴더
- `dataset.py` : 데이터셋을 만들고 augmentation을 정의하는 파일
- `loss.py` : loss를 정의하는 파일
- `model.py` : 모델 architecture를 정의
- `train.py` : dataset, loss, model 등 학습에 필요한 모듈을 불러오고 학습을 진행하고 모델을 저
- `inference.py` : 저장된 모델을 불러와서 eval 데이터로 제출 파일을 만드는 파일 (하나의모델 or 앙상블)
- `model/` : 모델들이 저장되는 폴더
- `output/` : 제출할 csv 파일이 저장되는 폴더

리더보드에 제출할 모델을 재현할 방법

```bash
~# python train.py --augmentation=CustomAugmentation --dataset=MaskSplitByProfileDataset --epochs=8 --model=EffB4Model
```

#### Architecture3 : vgg19

-vggtrain.ipynb : 전체 코드 파일
실제 앙상블에 사용한 vgg19_fold_8_0.803.pt 재현 불가
vgg19_fold_7_0.804.pt 또는 vgg19_fold_8_0.806.pt 사용
