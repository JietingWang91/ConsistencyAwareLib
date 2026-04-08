# Basic example for using ConsistencyAwareLib

from evaluation_index.baseline import MI, MIq, NMI, NMIq, NVIq, VI, VIq
from evaluation_index.adjusted import AMI, AMIq, AVIq, SG, SMI, SMIq, SVI, SVIq



# Two partitions (ground truth vs. predicted clustering)
labels_true =[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
labels_pred = [1, 1, 0, 0, 0, 2, 2, 2, 1, 1]

# 1) Baseline (uncorrected) metrics
mi  = MI(labels_true, labels_pred)
miq = MIq(labels_true, labels_pred, 2)
nmi = NMI(labels_true, labels_pred, method="sum")
nmiq = NMIq(labels_true, labels_pred, 2, method="sum")
nviq = NVIq(labels_true, labels_pred, 2)
vi = VI(labels_true, labels_pred)
viq = VIq(labels_true, labels_pred, 2)

# 2) Adjusted metrics
ami = AMI(labels_true, labels_pred, model="perm", method="sum", sided='two-sided')
amiq = AMIq(labels_true, labels_pred, 2, method="sum")
aviq = AVIq(labels_true, labels_pred, 2)
sg = SG(labels_true, labels_pred)
smi = SMI(labels_true, labels_pred)
smiq = SMIq(labels_true, labels_pred, 2)
svi = SVI(labels_true, labels_pred)
sviq = SVIq(labels_true, labels_pred, 2)

print("MI:", mi)
print("MIq:", miq)
print("NMI:", nmi)
print("NMIq:", nmiq)
print("NVIq:", nviq)
print("VI:", vi)
print("VIq:", viq)
print("AMI:", ami)
print("AMIq:", amiq)
print("AVIq:", aviq)
print("SG:", sg)
print("SMI:", smi)
print("SMIq:", smiq)
print("SVI:", svi)
print("SVIq:", sviq)

