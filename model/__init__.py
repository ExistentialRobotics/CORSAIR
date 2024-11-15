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

import logging
import model.simpleunet as simpleunets
import model.resunet as resunets
import model.fc as fc
import model.resnet as resnet

MODELS = []


def add_models(module):
    MODELS.extend([getattr(module, a) for a in dir(module) if "Net" in a or "MLP" in a])


add_models(simpleunets)
add_models(resunets)
add_models(fc)


def load_model(name):
    """Creates and returns an instance of the model given its class name."""
    # Find the model class from its name
    all_models = MODELS
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        logging.info(f"Invalid model index. You put {name}. Options are:")
        # Display a list of valid model names
        for model in all_models:
            logging.info("\t* {}".format(model.__name__))
        return None
    NetClass = mdict[name]

    return NetClass
