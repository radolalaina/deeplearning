import glob
import numpy
import CNNTransform

resnet = CNNTransform.resnet18()

def dist_prob(a, x):
    d = numpy.exp(((a - x)**2).sum(1))
    return d / d.sum()

class DeepMatch():
    def __init__(self, vector_dir="vectors"):
        files = glob.glob(vector_dir  + "/*.csv")
        labels = []
        V = []

        for f in files:
            print("Loading: ", f)
            labels.append(f.split("/")[-1].split(".")[0])
            V.append(numpy.loadtxt(f, delimiter=","))

        self.V = numpy.vstack(V)
        self.labels = numpy.hstack(labels)

    def predict(self, X):
        T = resnet.transform(X)

        # similarity
        D = [dist_prob(self.V, x).ravel() for x in T.detach().numpy()]

        p = []
        for d in D:
            idx = d.argsort()[:5]
            p.append(dict(zip(self.labels[idx], d[idx].tolist())))
        print(p)
        return p


if __name__ == '__main__':
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-i", "--image", dest="image",
                      help="Path to image")
