from loguru import logger
import onnxruntime as ort

import util
import config as conf


def main(conf):
    torch_model = util.load_backbone(conf)
    torch_model.eval()
    for param in torch_model.parameters():
        param.requires_grad = False
    logger.info('finish loading torch model')
    onnx_path = util.convert_onnx(torch_model, conf)
    logger.info('finish convert torch to onnx model')
    onnx_model = ort.InferenceSession(onnx_path)
    sess_option, onnx_optimize_path = util.optimize_graph(conf)
    onnx_graph_optimize = ort.InferenceSession(
        onnx_path, sess_option
    )
    onnx_quantize_model = util.quantize_onnx(onnx_path, conf)
    onnx_op_quantize_model = util.quantize_onnx(onnx_optimize_path, conf)
    util.test_time(
        onnx_model, onnx_graph_optimize,
        onnx_quantize_model, onnx_op_quantize_model,
        conf=conf
    )
    pass


if __name__ == '__main__':
    main(conf)
