import evaluation_index
from evaluation_index.adjusted import PureAccuracy

# Two partitions
labels_true = [0, 0, 1, 1, 2, 2]
labels_pred = [0, 1, 1, 1, 2, 0]

# Baseline metrics
mi  = evaluation_index.baseline.MI(labels_true, labels_pred)
nmi = evaluation_index.baseline.NMI(labels_true, labels_pred, method="sum")

# Adjusted metrics
ami = evaluation_index.adjusted.AMI(labels_true, labels_pred)

# Accuracy corrections (require labels mapped to 1..K)
def remap_to_1k(labels):
    uniq = sorted(set(labels))
    mp = {c: i + 1 for i, c in enumerate(uniq)}
    return [mp[x] for x in labels], len(uniq)

true_1k, Kt = remap_to_1k(labels_true)
pred_1k, Kp = remap_to_1k(labels_pred)
K = max(Kt, Kp)

pa = PureAccuracy(predicted_labels=pred_1k, true_labels=true_1k, num_classes=K)

print("MI:", mi)
print("NMI:", nmi)
print("AMI:", ami)
print("PureAccuracy:", pa)