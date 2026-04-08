from evaluation_index.baseline.NVIq import NVIq

def AVIq(labels1, labels2, q):
    #AVI_q is an adjusted similarity measure(with a maximum value of 1 and a value of 0 under randomness) whereas VI_q is a distance metric.
    # By using 1 - NVI_q, the normalized distance can be converted into an adjusted similarity.
    nviq=NVIq(labels1, labels2, q)
    aviq = 1 - nviq
    return aviq