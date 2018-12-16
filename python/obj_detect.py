import darknet as dn
from tqdm import tqdm
from matplotlib.image import imread

if __name__ == "__main__":
#    dn.set_gpu(0)
    net = dn.load_net(str.encode("data/yolov3-tiny.cfg"), str.encode("data/yolov3-tiny.weights"), 0)
    meta = dn.load_meta(str.encode("data/coco.data"))
    r = []
    # for speed benchmark
    img, _ = dn.array_to_image(imread("data/dog.jpg")) 
    r = dn.detect(net, meta, img, thresh=.2)
