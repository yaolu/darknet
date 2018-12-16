import darknet as dn
from tqdm import tqdm
from matplotlib.image import imread

if __name__ == "__main__":
    dn.set_gpu(0)
    net = dn.load_net(str.encode("data/yolov3-tiny.cfg"), str.encode("data/yolov3-tiny.weights"), 0)
    meta = dn.load_meta(str.encode("data/coco.data"))
    r = []
    # for speed benchmark
    for _ in tqdm(range(10)):
        r = dn.detect(net, meta, imread("data/dog.jpg"),thresh=.2)
    print(r)
