import argparse, glob, cv2, numpy as np
from symbol.symbol_factory import get_symbol
import mxnet as mx

data_size = 512
batchSize = 1
mean_rgb = [123.68, 116.779, 103.939]

def loadObjectDetector(model_path, epoch):
    # load SSD model
    net = get_symbol('vgg16_reduced', data_size, num_classes=1, nms_thresh=0.5, force_nms=True, nms_topk=400)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, epoch)
    # use cpu instead of gpu
    model = mx.mod.Module(net, label_names=None)
    model.bind(data_shapes=[('data', (batchSize, 3, data_size, data_size))])
    model.set_params(arg_params, aux_params, allow_missing=True)
    return model

def detectObjects(model, image):
    image = cv2.resize(image, (data_size, data_size))
    image = image - mean_rgb
    image = np.swapaxes(image, 0, 2)
    image = np.swapaxes(image, 1, 2)
    # load image into mxnet ndarray and then run detection
    ndarray_iter = mx.io.NDArrayIter(data=[mx.nd.array([image])] , batch_size= 1, last_batch_handle= 'pad')
    detections = model.predict(ndarray_iter).asnumpy()
    # only keeping class 0, if you want multi-class you'll need to modify!!
    detections = detections[np.where(detections[:,0]==0)]
    return detections

def showDetections(image, detections):
    rows, cols, d = image.shape
    for detection in detections:
        bbox = [detection[2]*cols, detection[3]*rows, detection[4]*cols, detection[5]*rows]
        bbox = list(map(int, bbox))
        cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),5)
    return image

model = loadObjectDetector('/Users/murrdl/nets/building_detector/ssd_vgg16_reduced_512', 100)

for f in glob.glob('*.jpg'):
    print(f)
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detectObjects(model, image.copy())
    print(len(detections),'objects detected')
    image_with_detections = showDetections(image.copy(), detections)
    cv2.imwrite(f[:-4]+'_det.jpg', cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR))
