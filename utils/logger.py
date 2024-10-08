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

import os
import time


class Logger:
    def __init__(self, log_dir, log_name):
        self.path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        open(self.path, "w").close()

    def log(self, txt):
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print("{} {}".format(t, txt))
        if isinstance(txt, str):
            with open(self.path, "a") as f:
                f.write("{} {}".format(t, txt) + "\n")
