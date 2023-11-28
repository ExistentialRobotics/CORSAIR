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
- Download pre-processed Scan2CAD data from [here](https://drive.google.com/file/d/1dOR4Y13rBxmxS-sIF97eMWnjf81Toaqd/view?usp=sharing) and extract it to `data/Scan2CAD_pc`.
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
  --cache-dir CACHE_DIR
                        Path to load / save the result of registration.
  --register-gt         Registering gt CAD model
  --device {cuda,cpu}   Device to use for evaluation
  --ignore-cache        Ignore cached results
```

For example,
```shell
python evaluation.py --shapenet-pc15k-root data/ShapeNetCore.v2.PC15k --scan2cad-pc-root data/Scan2CAD_pc --scan2cad-annotation-root data/Scan2CAD_annotations --category chair --checkpoint ckpts/scannet_pose_chair_best --cache-dir data/cache_pose_best --register-gt --device cuda
```
will evaluate the model on the chair category and register the gt CAD model using the GPU.

![](assets/gui.png)

- Press Left/Right to navigate through the results.
- Drag the mouse to rotate the object.

# Metrics

## Retrival
|       Checkpoint        | Precision@10% | Top1-CD | Mean Rotation Error (wo/w sym.) | Mean Translation Error (wo/w sym.) |
| :---------------------: | :-----------: | :-----: | :-----------------------------: | :--------------------------------: |
| scannet_pose_chair_best |    22.55%     |  0.17   |  $39.17^\circ$ / $38.74^\circ$  |            0.28 / 0.27             |
|                         |               |         |                                 |                                    |
|                         |               |         |                                 |                                    |
|                         |               |         |                                 |                                    |
|                         |               |         |                                 |                                    |

## Registration
- RRE: relative rotation error
- RTE: relative translation error

|       Checkpoint        | Registration Target | Sym.  |   Mean RRE    | RRE$\le 5^\circ$ | RRE$\le 15^\circ$ | RRE$\le 45^\circ$ | Mean RTE | RTE$\le 0.02$ | RTE$\le 0.05$ | RTE$\le 0.10$ | RTE$\le 0.15$ |
| :---------------------: | :-----------------: | :---: | :-----------: | :--------------: | :---------------: | :---------------: | :------: | :-----------: | :-----------: | :-----------: | :-----------: |
| scannet_pose_chair_best |   Top1-Prediction   |   N   | $39.17^\circ$ |      6.64%       |      54.78%       |      80.36%       |   0.28   |     0.20%     |     3.93%     |    20.95%     |    43.61%     |
|                         |                     |   Y   | $38.74^\circ$ |      9.87%       |      59.82%       |      81.17%       |   0.27   |     0.30%     |     4.53%     |    23.36%     |    47.33%     |
|                         |    G.T. Matching    |       |               |                  |                   |                   |          |               |               |               |               |
|                         |                     |       |               |                  |                   |                   |          |               |               |               |               |
|                         |                     |       |               |                  |                   |                   |          |               |               |               |               |
|                         |                     |       |               |                  |                   |                   |          |               |               |               |               |
|                         |                     |       |               |                  |                   |                   |          |               |               |               |               |
|                         |                     |       |               |                  |                   |                   |          |               |               |               |               |
