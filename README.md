
# Deepfake detection

 
This project is my research project for my second-year master IISC Data Science and Machine Learning. 
I have tested two models with différente dataset.



## Run Locally

Clone the project

```bash
  git clone https://github.com/Uhouhzam/Deepfake-Detection.git
```

Go to the project directory

```bash
  cd Deepfake-Detection
```

Training the model Xception and save the pre-trained weight in result_root

```bash
  py TrainXception.py data_root result_root
```
Test the model Xception with pre-trained weight in weight_root, 
draw the ROC and accuracy curve, save curves and the result prediction as csv in result_root.

```bash
  py PredictionXception.py data_root weight_root/pre-tained_weight result_root
```
Training the model Meso-4 and save the pre-trained weight in result_root

```bash
  py TrainMeso4.py data_root result_root
```
Test the model Meso-4 with pre-trained weight in weight_root, 
draw the ROC and accuracy curve, save curves and the result prediction as csv in result_root.

```bash
  py TestMeso4.py data_root weight_root/pre-tained_weight result_root
```
## Author

- [@Mazhou HU](https://github.com/Uhouhzam)


## Related

My programme is inspired from this two project respectively from DariusAf and i3p9.

[MesoNet](https://github.com/DariusAf/MesoNet)

[deepfake-detection-with-xception](https://github.com/i3p9/deepfake-detection-with-xception)



## Reference

#### Meso-4 Model

Darius Afchar et al. “Mesonet: a compact facial video forgery detection network”. In: 2018
IEEE International Workshop on Information Forensics and Security (WIFS). IEEE.
2018, pp. 1–7.

#### Xception Model
François Chollet. “Xception: Deep Learning
with Depthwise Separable Convolutions”. In:
CoRR abs/1610.02357 (2016). arXiv: 1610 .
02357. url: http://arxiv.org/abs/1610.
02357.



## Data set used

| Dataset             | Number training set |        Number test set                                                      |
| ----------------- | ----------------------|------------------------------------------- |
| StarGAN |2024  |3614|
| AttGAN | 2024 |3971|
| GDWCT | 2024 |1333|
| CelebA | 2024|3614/3971/1333|
| Autre(real/fake) | 5103/5103 |2845|


