from loguru import logger
import onnxruntime as ort

import util
import config as conf


def main(conf):
    torch_model = util.load_backbone(conf)
    logger.info('finish loading torch model')
    onnx_path = util.convert_onnx(torch_model, conf)
    logger.info('finish convert torch to onnx model')
    onnx_model = ort.InferenceSession(onnx_path)
    onnx_graph_optimize = ort.InferenceSession(
        onnx_path, util.optimize_graph()
    )
    onnx_quantize_path = util.quantize_onnx(torch_model, conf)
    onnx_quantize = ort.InferenceSession(
        onnx_quantize_path
    )
    util.test_time(onnx_model, onnx_graph_optimize, onnx_quantize, conf=conf)
    pass


if __name__ == '__main__':
    main(conf)
