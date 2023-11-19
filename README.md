CORSAIR
=======

Repo for paper [CORSAIR: Convolutional Object Retrieval and Symmetry-AIded Registration](https://ieeexplore.ieee.org/document/9636347).

# Setup

## Python Environment

```shell
git clone --recursive --branch main https://github.com/ExistentialRobotics/CORSAIR.git
pipenv lock --verbose
pipenv sync --verbose
pipenv shell
cd deps/MinkowskiEngine  # 0.5.5
pip install . --verbose
```

## Demo Data

- Download ShapeNetCore.v2.PC15k from [here](https://drive.google.com/file/d/1myIBzh8_Ja5gXoz6MiSAaZWXe4BQ68yB/view?usp=sharing) and extract it to `data/ShapeNetCore.v2.PC15k`.
- Download pre-processed Scan2CAD data from [here](https://drive.google.com/file/d/1dOR4Y13rBxmxS-sIF97eMWnjf81Toaqd/view?usp=sharing) and extract it to `data/Scan2CAD`.
- Download Scan2CAD annotation data from [here](https://drive.google.com/file/d/1zPajN8FyOJtdLNdam_Dtw9SHmq5GaVs9/view?usp=sharing) and extract it to `data/Scan2CAD_annotations`.
- Download the pre-trained models from [here]()
