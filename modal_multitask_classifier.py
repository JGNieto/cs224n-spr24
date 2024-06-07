import modal
from modal import Image
import os
import multitask_classifier
import random
import string

current_directory = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_NAME = None

# NOTE: It is possible that this file is run multiple times by Modal for a single
# experiment, and EXPERIMENT_NAME will change between runs. Therefore, we CANNOT
# use the value of EXPERIMENT_NAME in this file, only the definitive value in
# multitask_classifier.py. (TL;DR, don't print EXPERIMENT_NAME in this file)
if not EXPERIMENT_NAME:
    EXPERIMENT_NAME = ''.join(random.choices(string.ascii_lowercase, k=6))

LOCAL_DATA_DIR = os.path.join(current_directory, 'data')
REMOTE_DATA_DIR = "/root/jbelle/data"
VOLUME_NAME = "jbelle-data"
VOLUME_PATH = "/vol/jbelle"
REMOTE_PREDICTIONS_DIR = f"{VOLUME_PATH}/{EXPERIMENT_NAME}/predictions"
REMOTE_OUTPUT_DIR = f"{VOLUME_PATH}/{EXPERIMENT_NAME}/output"
GPU = "A10G"

volume = modal.Volume.from_name(VOLUME_NAME)

packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "tqdm",
    "requests",
    "importlib-metadata",
    "filelock",
    "scikit-learn",
    "tokenizers",
    "explainaboard_client",
    "numpy"
]

class Params:
    def __init__(self, params):
        self.sst_train = params.get("sst_train")
        self.sst_dev = params.get("sst_dev")
        self.sst_test = params.get("sst_test")
        self.para_train = params.get("para_train")
        self.para_dev = params.get("para_dev")
        self.para_test = params.get("para_test")
        self.sts_train = params.get("sts_train")
        self.sts_dev = params.get("sts_dev")
        self.sts_test = params.get("sts_test")
        self.seed = params.get("seed")
        self.epochs = params.get("epochs")
        self.fine_tune_mode = params.get("fine_tune_mode")
        self.sst_dev_out = params.get("sst_dev_out")
        self.sst_test_out = params.get("sst_test_out")
        self.para_dev_out = params.get("para_dev_out")
        self.para_test_out = params.get("para_test_out")
        self.sts_dev_out = params.get("sts_dev_out")
        self.sts_test_out = params.get("sts_test_out")
        self.batch_size = params.get("batch_size")
        self.hidden_dropout_prob = params.get("hidden_dropout_prob")
        self.last_dropout_prob = params.get("last_dropout_prob")
        self.lr = params.get("lr")
        self.pcgrad = params.get("pcgrad")
        self.dora = params.get("dora")
        self.lora = params.get("lora")
        self.l1l2 = params.get("l1l2")
        self.eval = params.get("eval")
        self.parallel = params.get("parallel")
        self.task = params.get("task")
        self.load = params.get("load")
        self.decay = params.get("decay")
        self.early_stop = params.get("early_stop")
        self.nickname = params.get("nickname")
        self.output = params.get("output")
        self.one_at_a_time = params.get("one_at_a_time")
        self.smart_lambda = params.get("smart_lambda")
        self.save_losses = params.get("save_losses")

params = {
    "sst_train": f"{REMOTE_DATA_DIR}/ids-sst-train.csv",
    "sst_dev": f"{REMOTE_DATA_DIR}/ids-sst-dev.csv",
    "sst_test": f"{REMOTE_DATA_DIR}/ids-sst-test-student.csv",
    "para_train": f"{REMOTE_DATA_DIR}/quora-train.csv",
    "para_dev": f"{REMOTE_DATA_DIR}/quora-dev.csv",
    "para_test": f"{REMOTE_DATA_DIR}/quora-test-student.csv",
    "sts_train": f"{REMOTE_DATA_DIR}/sts-train.csv",
    "sts_dev": f"{REMOTE_DATA_DIR}/sts-dev.csv",
    "sts_test": f"{REMOTE_DATA_DIR}/sts-test-student.csv",
    "seed": 11711,
    "sst_dev_out": f"{REMOTE_PREDICTIONS_DIR}/sst-dev-output.csv",
    "sst_test_out": f"{REMOTE_PREDICTIONS_DIR}/sst-test-output.csv",
    "para_dev_out": f"{REMOTE_PREDICTIONS_DIR}/para-dev-output.csv",
    "para_test_out": f"{REMOTE_PREDICTIONS_DIR}/para-test-output.csv",
    "sts_dev_out": f"{REMOTE_PREDICTIONS_DIR}/sts-dev-output.csv",
    "sts_test_out": f"{REMOTE_PREDICTIONS_DIR}/sts-test-output.csv",
    "batch_size": 8,
    "fine_tune_mode": "full-model",
    "hidden_dropout_prob": 0.3,
    "last_dropout_prob": 0.5,
    "lr": 1e-5,
    "pcgrad": False,
    "dora": False,
    "lora": False,
    "l1l2": False,
    "parallel": False,
    "task": "multi",
    "epochs": 10,
    "early_stop": 4,
    "decay": 0.01,
    "nickname": "",
    "output": REMOTE_OUTPUT_DIR,
    "one_at_a_time": True,
    "smart_lambda": 10,
    "save_losses": True,
    "eval": True,
    "load": os.path.join(VOLUME_PATH, "vmlnmg"),
}

image = (
        Image.debian_slim()
        .pip_install(*packages, gpu=GPU)
        )

app = modal.App()

@app.function(image=image,
              mounts=[modal.Mount.from_local_dir(LOCAL_DATA_DIR, remote_path=REMOTE_DATA_DIR)],
              gpu=GPU,
              timeout=60*60*5,
              volumes={VOLUME_PATH: volume},
              )
def run_multitask():
    params_obj = Params(params)
    nick = EXPERIMENT_NAME
    params_obj.nickname = nick
    try:
        multitask_classifier.run(params_obj)
    except Exception as e:
        print(f"Error: {e}")
        print("Saving volume...")
        volume.commit()
        raise
    print(f"Finished running multitask_classifier experiment {nick}. Writing to modal...")
    volume.commit()

