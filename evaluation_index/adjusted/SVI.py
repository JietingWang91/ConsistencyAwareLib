from evaluation_index.adjusted.SG import SG

def SVI(labels1, labels2):
# SMI = SVI = SG
    svi=SG(labels1, labels2)
    return svi