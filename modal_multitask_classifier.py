import modal
from modal import Image
import os
import multitask_classifier

current_directory = os.path.dirname(os.path.abspath(__file__))

LOCAL_DATA_DIR = os.path.join(current_directory, 'data')
REMOTE_DATA_DIR = "/root/jbelle/data"
REMOTE_PREDICTIONS_DIR = "/root/jbelle/predictions"
GPU = "T4"

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
    "hidden_dropout_prob": 0.3,
    "last_dropout_prob": 0.4,
    "lr": 5e-5,
    "pcgrad": False,
    "dora": False,
    "l1l2": False,
    "eval": False,
    "parallel": False,
    "task": "multi",
    "load": None,
    "decay": 0,
    # "decay": 0.01
}

image = (
        Image.debian_slim()
        .pip_install(*packages, gpu=GPU)
        )

app = modal.App()

@app.function(image=image,
              mounts=[modal.Mount.from_local_dir(LOCAL_DATA_DIR, remote_path=REMOTE_DATA_DIR)],
              gpu=GPU,
              timeout=60*60*3 # 3 hours
              )
def run_multitask():
    params_obj = Params(params)
    multitask_classifier.run(params_obj)

