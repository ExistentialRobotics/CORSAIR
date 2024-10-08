'''
   Copyright 2024 Qiaojun Feng, Sai Jadhav, Tianyu Zhao, Zhirui Dai, K. M. Brian Lee, Nikolay Atanasov, UC San Diego. 

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=str, nargs="+", required=True)
    parser.add_argument("--n-models", type=int, default=100)
    parser.add_argument("--n-poses-per-model", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    table = pd.DataFrame(
        columns=[
            "category",
            "rre_005_sym",
            "rre_015_sym",
            "rre_045_sym",
            "rre_005_ransac",
            "rre_015_ransac",
            "rre_045_ransac",
            "rte_002_sym",
            "rte_005_sym",
            "rte_010_sym",
            "rte_015_sym",
            "rte_002_ransac",
            "rte_005_ransac",
            "rte_010_ransac",
            "rte_015_ransac",
        ]
    )
    for category in args.categories:
        postfix = f"shapenet-seed{args.random_seed}-{category}-{args.n_models}-{args.n_poses_per_model}"
        df = pd.read_csv(f"results-{postfix}.csv")

        rre_005_sym = (df["rre_sym"] <= np.deg2rad(5)).sum() / len(df)
        rre_015_sym = (df["rre_sym"] <= np.deg2rad(15)).sum() / len(df)
        rre_045_sym = (df["rre_sym"] <= np.deg2rad(45)).sum() / len(df)
        rre_005_ransac = (df["rre_ransac"] <= np.deg2rad(5)).sum() / len(df)
        rre_015_ransac = (df["rre_ransac"] <= np.deg2rad(15)).sum() / len(df)
        rre_045_ransac = (df["rre_ransac"] <= np.deg2rad(45)).sum() / len(df)
        rte_002_sym = (df["rte_sym"] <= 0.02).sum() / len(df)
        rte_005_sym = (df["rte_sym"] <= 0.05).sum() / len(df)
        rte_010_sym = (df["rte_sym"] <= 0.10).sum() / len(df)
        rte_015_sym = (df["rte_sym"] <= 0.15).sum() / len(df)
        rte_002_ransac = (df["rte_ransac"] <= 0.02).sum() / len(df)
        rte_005_ransac = (df["rte_ransac"] <= 0.05).sum() / len(df)
        rte_010_ransac = (df["rte_ransac"] <= 0.10).sum() / len(df)
        rte_015_ransac = (df["rte_ransac"] <= 0.15).sum() / len(df)
        table.loc[len(table)] = [
            category,
            rre_005_sym,
            rre_015_sym,
            rre_045_sym,
            rre_005_ransac,
            rre_015_ransac,
            rre_045_ransac,
            rte_002_sym,
            rte_005_sym,
            rte_010_sym,
            rte_015_sym,
            rte_002_ransac,
            rte_005_ransac,
            rte_010_ransac,
            rte_015_ransac,
        ]
    print(table.transpose())


if __name__ == "__main__":
    main()
