import modal
from modal import Image
import os
import multitask_classifier
import random
import string

current_directory = os.path.dirname(os.path.abspath(__file__))

EXPERIMENT_NAME = None

if not EXPERIMENT_NAME:
    EXPERIMENT_NAME = ''.join(random.choices(string.ascii_lowercase, k=6))

print(f"Experiment name: {EXPERIMENT_NAME}")

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
        self.l1l2 = params.get("l1l2")
        self.eval = params.get("eval")
        self.parallel = params.get("parallel")
        self.task = params.get("task")
        self.load = params.get("load")
        self.decay = params.get("decay")
        self.early_stop = params.get("early_stop")
        self.nickname = params.get("nickname")
        self.output = params.get("output")

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
    "epochs": 10,
    "fine_tune_mode": "full-model",
    "sst_dev_out": f"{REMOTE_PREDICTIONS_DIR}/sst-dev-output.csv",
    "sst_test_out": f"{REMOTE_PREDICTIONS_DIR}/sst-test-output.csv",
    "para_dev_out": f"{REMOTE_PREDICTIONS_DIR}/para-dev-output.csv",
    "para_test_out": f"{REMOTE_PREDICTIONS_DIR}/para-test-output.csv",
    "sts_dev_out": f"{REMOTE_PREDICTIONS_DIR}/sts-dev-output.csv",
    "sts_test_out": f"{REMOTE_PREDICTIONS_DIR}/sts-test-output.csv",
    "batch_size": 8,
    "hidden_dropout_prob": 0.5,
    "last_dropout_prob": 0.6,
    "lr": 2e-5,
    "pcgrad": False,
    "dora": False,
    "l1l2": False,
    "eval": False,
    "parallel": False,
    "task": "multi",
    "load": None,
    "early_stop": -1,
    "decay": 0.01,
    "nickname": "",
    "output": REMOTE_OUTPUT_DIR,
}

image = (
        Image.debian_slim()
        .pip_install(*packages, gpu=GPU)
        )

app = modal.App()

@app.function(image=image,
              mounts=[modal.Mount.from_local_dir(LOCAL_DATA_DIR, remote_path=REMOTE_DATA_DIR)],
              gpu=GPU,
              timeout=60*60*3, # 3 hours
              volumes={VOLUME_PATH: volume},
              )
def run_multitask():
    params_obj = Params(params)
    params_obj.nickname = EXPERIMENT_NAME
    multitask_classifier.run(params_obj)
    print("Finished running multitask_classifier. Writing to modal...")
    volume.commit()

