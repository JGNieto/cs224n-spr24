from datasets import load_multitask_data

# Load the dataset
sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
    load_multitask_data("data/ids-sst-dev.csv","data/quora-dev.csv","data/sts-dev.csv",split='dev')


map = {}

for datapoint in sst_dev_data:
    sent, label, sent_id = datapoint
    map[sent_id] = label

off_by = [0] * 5

# Load the output of the model
with open('ensembles/emnzai-ysfdjs-wbpjwk-xstxpb-ugcqti-bhqida-zymtvr/sst-dev-output.csv', 'r') as f:
    f.readline() # ignore the header
    for line in f:
        parts = line.strip().split(',')
        example_id = parts[0].strip()
        prediction = int(parts[1].strip())
        if example_id not in map:
            raise ValueError("Example ID not found in the dataset")

        off = abs(prediction - map[example_id])
        off_by[off] += 1

for i in range(5):
    print(f"Off by {i}: {off_by[i]}")
