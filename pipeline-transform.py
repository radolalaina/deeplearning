import image_loader
import CNNTransform
import numpy

output_dir = "vectors"

def save(X, fname):
    output_file = "%s/%s.csv"%(output_dir, fname)
    numpy.savetxt(output_file, X.detach().numpy(), delimiter=",")
    print("Saved: ", output_file)

resnset = CNNTransform.resnet18()

for image, path, area in image_loader.load_json("/home/ikarandoha/Documents/projetMaster2/datacow/train/holstein"):
    if len(path) > 0:
        i = 0
        i += 1
        filename = path.split("/")[-1].split(".")[0] + "-" + str(i)
        numpy.savetxt("localization/%s.csv"%filename, area, delimiter=",")
        X = resnset.transform(image[:3].reshape(1,3,128,128))
        save(X, filename)
