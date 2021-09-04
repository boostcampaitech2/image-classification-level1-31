# Boost Camp 2기 31조 Level 1 - Mask Image Classification Competition

### Getting Started

1. `tf_efficientnet_b4_ns` 폴더에서 앙상블에 사용할 architecture-2 모델 만들기. 방법은 아래 설명되어 있습니다.
2. `vggtrain.ipynb` 에서 앙상블에 사용할 architecture-3 모델 만들기.
3. `swin_base_patch4_window12_384_and_tf_efficientnet_b4_ns` 폴더에서 앙상블에 사용할 architecture-1 모델 만들기. 방법은 아래에 설명되어 있습니다.
4. architecture 1,2,3을 `inference_test.py` 의 CFG에서 지정한 폴더(default : data/ensemble)에 한 곳에 모으고`inference_test.py` 실행으로 `submmission.csv ` 생성

### Installation

---

All requirements should be detailed in requirements.txt

```
pip install -r requirements.txt
```

### Archive contents

---

#### Architecture1 : swin_base_patch4_window12_384

- swin_base_patch4_window12_384_and_tf_efficientnet_b4_ns : contains original code, trained models etc

```
swin_base_patch4_window12_384_and_tf_efficientnet_b4_ns
|-- data_utils
|   |-- data_loaders.py
|   |-- datasets.py
|   `-- make_df_sep_val_trn.py
|-- model
|   |-- loss.py
|   `-- models.py
|-- inference_test.py
|-- train.py
|-- tools
|   |-- face_crop.py
|   `-- face_crop_eval.py
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
- `tools` : 이미지 데이터를 얼굴 부분만 잘라주는 tool

##### 리더보드에 제출할 모델을 재현할 방법

```
1. tools의 face_crop.py , face_crop_eval.py 실행
2. make_df_sep_val_trn.py 실행으로 train / test 용 csv split
3-1. CFG의 'model_arch'를 swin_base_patch4_window12_384로 설정하고 train.py 실행
3-2. data/saved_architecture_1 에서 swin_base_patch4_window12_384_fold_0_4_0.788.pt, swin_base_patch4_window12_384_fold_1_9_0.821.pt를 앙상블 할 폴더에 저장
4-1. CFG의 'model_arch'를 tf_efficientnet_b4_ns로 설정하고 train.py 실행
4-2. data/saved_architecture_1 에서 tf_efficientnet_b4_ns_fold_0_7_0.833.pt를 앙상블 할 폴더에 저장
5. 트레이닝된 model을 모아 inferece_test.py를 통해 앙상블 후 submission파일로 저장

참고 : train.py와 inference_test.py 의 CFG를 통해 자신의 환경에 맞게 바꿔야함
```

#### Architecture2 : tf_efficientnet_b4_ns

```
tf_efficientnet_b4_ns
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

- `tf_efficientnet_b4_ns/` : tf_efficientnet_b4_ns모델을 만드는 코드가 들어있는 폴더
- `dataset.py` : 데이터셋을 만들고 augmentation을 정의하는 파일
- `loss.py` : loss를 정의하는 파일
- `model.py` : 모델 architecture를 정의
- `train.py` : dataset, loss, model 등 학습에 필요한 모듈을 불러오고 학습을 진행하고 모델을 저
- `inference.py` : 저장된 모델을 불러와서 eval 데이터로 제출 파일을 만드는 파일 (하나의모델 or 앙상블)
- `model/` : 모델들이 저장되는 폴더
- `output/` : 제출할 csv 파일이 저장되는 폴더

리더보드에 제출할 모델을 재현할 방법

```bash
~# python train.py --augmentation=CustomAugmentation --dataset=MaskSplitByProfileDataset --epochs=8 --model=EffB4Model --data_dir='데이터 경로를 입력해주세요'
```

#### Architecture3 : vgg19

```
vgg19
|--saved_model
|--vggtrain.ipynb
|--new_train.csv
|--new_valid.csv
```

- `vggtrain.ipynb` : 전체 코드 파일

```
실제 앙상블에 사용한 vgg19_fold_8_0.803.pt 재현 불가
saved_model에 저장된 vgg19_fold_7_0.804.pt 또는 vgg19_fold_8_0.806.pt 사용
```

##### Reference
- https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-train-amp-aug
- https://www.kaggle.com/khyeh0719/pytorch-efficientnet-baseline-inference-tta
- https://www.kaggle.com/piantic/train-cassava-starter-using-various-loss-funcs
- https://stages.ai/competitions/74/discussion/talk/post/436
- https://stages.ai/competitions/74/discussion/talk/post/475
- https://stages.ai/competitions/74/discussion/talk/post/493
