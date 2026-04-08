from evaluation_index.adjusted.SG import SG

def SMI(labels1, labels2):
# SMI = SVI = SG
    smi=SG(labels1, labels2)
    return smi