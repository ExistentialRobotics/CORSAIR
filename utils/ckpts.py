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

import torch
import os


def load_checkpoint(
    model,
    embedding,
    optimizer,
    scheduler,
    path,
):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    embedding.load_state_dict(checkpoint["embedding_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    epoch = checkpoint["epoch"]
    return model, embedding, optimizer, epoch


def save_checkpoint(model, embedding, optimizer, scheduler, epoch, save_dir, save_name):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir, save_name)
    if embedding == None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            path,
        )
    else:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "embedding_state_dict": embedding.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            path,
        )
