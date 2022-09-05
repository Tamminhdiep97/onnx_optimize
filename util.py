import os
from os.path import join as opj
import time

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152


def load_backbone(config):
    model_name = config.backbone_name
    model_dict = {
        'IR_50': IR_50(config.size),
        'IR_101': IR_101(config.size),
        'IR_152': IR_152(config.size),
        'IR_SE_50': IR_SE_50(config.size),
        'IR_SE_101': IR_SE_101(config.size),
        'IR_SE_152': IR_SE_152(config.size),
    }
    model = model_dict[model_name]
    del model_dict
    logger.info('create backbone: {}'.format(model_name))
    logger.info('input_layer: ')
    logger.info(list(model.input_layer.children()))
    logger.info('body_layer: ')
    logger.info(list(model.body.children()))
    logger.info('output_layer: ')
    logger.info(list(model.output_layer.children()))
    return model


def convert_onnx(model, config):
    input_size = int(config.size[0])
    dummy_input = torch.randn(
        1, 3, input_size, input_size, device=torch.device('cpu')
    )
    input_names = ['input_1']
    output_names = ['output_1']
    dynamic_axes = {
        'input_1': [0, 2, 3],
        'output_1': {
            0: 'output_1_variable_dim_0',
            1: 'output_1_variable_dim_1'
        },
    }
    model_name = '{}.onnx'.format(config.backbone_name.lower())
    onnx_folder = opj(
        os.getcwd(), 'weight', config.backbone_name.lower()
    )
    os.makedirs(onnx_folder, exist_ok=True)
    onnx_path = opj(onnx_folder, model_name)
    torch.onnx.export(
        model, dummy_input, onnx_path, verbose=True,
        opset_version=config.opset_onnx, input_names=input_names,
        output_names=output_names, dynamic_axes=dynamic_axes,
    )
    return onnx_path


def optimize_graph():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return sess_options


def quantize_onnx(onnx_model_path, config):
    model_quantize_name = '{}.quantize.onnx'.format(
        config.backbone_name.lower()
    )
    onnx_folder = opj(os.getcwd(), 'weight', config.backbone_name.lower())
    model_quantize_path = opj(onnx_folder, model_quantize_name)
    _ = quantize_dynamic(
        onnx_model_path, model_quantize_path, weight_type=QuantType.QUInt8
    )
    return ort.InferenceSession(model_quantize_path)


def test_time(*onnx_model, conf):
    for ith, model in enumerate(onnx_model):
        logger.info('model: {}'.format(ith))
        input_tensors = np.random.randn(
            conf.sample, 1, 3, 112, 112
        ).astype(
            np.float32
        )
        t_start = time.time()
        for tensor in tqdm(input_tensors):
            _ = model.run(
                None,
                {'input_1': tensor},
            )
        t_end = time.time()
        logger.info(
            'test model {} on {} samples take {} s'.format(
                ith, conf.sample, str(t_end-t_start)
            )
        )
        average_time = (t_end-t_start)/conf.sample
        logger.info('take average {}s for each sample'.format(average_time))
        time.sleep(10)
    pass
