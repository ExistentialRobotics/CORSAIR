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
