import cv2
import numpy as np
import onnx
import time
import onnxruntime
import matplotlib.pyplot as plt

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

providers_gpu = ['CUDAExecutionProvider']
providers_cpu = ['CPUExecutionProvider']



def normalize(data, mean, std):
    # transforms.ToTensor, transforms.Normalize
    if not isinstance(mean, np.ndarray):
        mean = np.array(mean)
    if not isinstance(std, np.ndarray):
        std = np.array(std)
    if mean.ndim == 1:
        mean = np.reshape(mean, (-1, 1, 1))
    if std.ndim == 1:
        std = np.reshape(std, (-1, 1, 1))
    _max = np.max(abs(data))
    _div = np.divide(data, _max)  # i.e. _div = data / _max
    _sub = np.subtract(_div, mean)  # i.e. arrays = _div - mean
    arrays = np.divide(_sub, std)  # i.e. arrays = (_div - mean) / std
    arrays = np.transpose(arrays, (2, 0, 1))
    return arrays


def find_object_bbox(mask_image):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = []
    areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areas.append(area)
    max_cnt_id = np.argmax(areas)
    x, y, w, h = cv2.boundingRect(contours[max_cnt_id])
    bbox.append(x)
    bbox.append(y)
    bbox.append(w)
    bbox.append(h)
    bbox.append(bbox)
    return bbox


def pred_onnx(model_file, image_file):
    m = onnx.load(model_file)
    onnx.checker.check_model(m, True)
    session = onnxruntime.InferenceSession(model_file, providers=providers_gpu)
    # sess_options = session.get_session_options()
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    input_shape = session.get_inputs()[0].shape
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), 1)
    src_image = image.copy()
    image = normalize(image, [0.5], [0.5])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    t0 = time.time()
    pred = session.run([output_name], {input_name: image})[0]
    t1 = time.time()
    print("output shape = {}".format(pred.shape))
    # print(pred)
    pred_shape = pred.shape
    if len(pred_shape) == 4:
        pred = np.transpose(pred, (2, 3, 0, 1))
        pred = pred[:, :, :, 0]
    else:
        pred = np.transpose(pred, (1, 2, 0))
    print("pred done, cost={}".format(t1 - t0))
    pred = pred * 255
    pred = np.squeeze(pred, axis=2)
    pred = pred.astype(np.uint8)
    bbox = find_object_bbox(pred)
    cv2.rectangle(src_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)
    rgb_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.show()


if __name__ == '__main__':
    im = './1.png'
    model = './output.onnx'
    pred_onnx(model, im)
