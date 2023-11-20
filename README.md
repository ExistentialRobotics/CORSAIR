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

# Usage

```text
Evaluate CORSAIR

options:
  -h, --help            show this help message and exit
  --shapenet-pc15k-root SHAPENET_PC15K_ROOT
                        Path to ShapeNetCore.v2.PC15k
  --scan2cad-pc-root SCAN2CAD_PC_ROOT
                        Path to Scan2CAD
  --scan2cad-annotation-root SCAN2CAD_ANNOTATION_ROOT
                        Path to Scan2CAD annotations
  --category {table,chair}
                        Category to evaluate
  --checkpoint CHECKPOINT
                        Path to the checkpoint
  --device {cuda,cpu}   Device to use for evaluation
  --ignore-cache        Ignore cached results
```

![](assets/gui.png)

- Press Left/Right to navigate through the results.
- Drag the mouse to rotate the object.

# Issues

The ground truth is wrong such that the error is higher than the paper and the visualization of `Closest CAD PC` 
presents mis-alignment. We will fix it soon.
