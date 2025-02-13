
import logging as log
import os
import sys
import traceback
from argparse import ArgumentParser, SUPPRESS
import numpy as np
from pathlib import Path
from openvino.runtime import InferRequest, Output
from openvino.runtime.utils.types import get_dtype

from concreteModel import ConcreteModel

verbosity = True
def error_handling(desc: str):
    """
    Error handler that prints description formatted with keyword arguments in case of exception
    :param desc: description for an error
    :return: decorator
    """

    def decorator(func):
        def try_except_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                exception_type = type(e).__name__
                log.error(f"The following error happened while {desc.format(**kwargs)}:\n[ {exception_type} ] {e}")
                global verbosity
                if verbosity:
                    traceback.print_tb(tb=e.__traceback__, file=sys.stdout)
                sys.exit(1)

        return try_except_func

    return decorator

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    
    args.add_argument("-m", "--model", help="Required. Path to a .xml file.", required=True, type=str, default=None)
    args.add_argument("-b", "--blob", help="Optional. Path to a .blob file. Would be compiled from --model file if doesn't set.", required=False, type=str, default=None)
    args.add_argument("-d", "--device", help="Optional. Device for reference, default is CPU", required=False, type=str)
    
    args.add_argument("-i", "--inputs", help="Required. Path to inputs file.", nargs='+', required=False, type=str, default=None)
    args.add_argument("-no", "--npu_outputs", help="Required. Path to npu's outputs file.", nargs='+', required=False, type=str, default=None)

    args.add_argument("-ip", "--input_precision", help="precision of input tensor", nargs='+', default="f32", type=str, required=False)
    args.add_argument("-op", "--output_precision", help="precision of output tensor", nargs='+', default="f32", type=str, required=False)
    
    args.add_argument('--dump', help='Enables ref and npu output .bin file dumping, defualt folder will be the root folder of --model,'
                      ' identify it by --output_dir/-out_dir', action='store_true', default=False)
    args.add_argument('--print', help='Enables ref and npu output show up', action='store_true', default=False)
    args.add_argument('--mat', help='Enables ref and npu output comparision diagram show up', action='store_true', default=False)
    args.add_argument("-out_dir", "--output_dir", help="Output Saved folder", nargs='+', type=str, required=False)
    
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    parsed_args = parser.parse_args()
    if parsed_args.output_dir is None:
        parsed_args.output_dir = os.path.dirname(parsed_args.model)

    return parsed_args

def get_model_name(model_path):
    return os.path.splitext(os.path.basename(model_path))[0]

def save_array_to_file(array, folder, model_name, tensor_name = None, device = None):
    path = os.path.join(folder, model_name)
    if tensor_name:
        tensor_name = tensor_name.replace('/', '-')
    path += f'_[{tensor_name}]' if tensor_name else ''
    path += f'_{device}' if device else ''
    path += str(array.dtype) + '.bin'

    array.tofile(path)
    log.info(f"Saved {tensor_name} to {path}")

def process_dict_or_list_args(args, keys):
    args_dict = {}
    for idx, arg in enumerate(args):
        if ':' in arg:
            key, value = arg.split(":")

            if key not in keys.keys():
                raise Exception(f"Unknown argument {key} in {args}")
            args_dict[key] = value
        else:
            args_dict[keys[idx]] = arg
    return args_dict

### 
# Model
### 

def get_combined_model(core, model, blob, device :str):   
    return ConcreteModel(core, model, blob, device)

###
# Infer
###

@error_handling('getting inference results for output: \'{output.any_name}\'')
def get_infer_results(infer_request: InferRequest, output: Output):
    return infer_request.get_tensor(output).data

def infer(model: ConcreteModel, inputs):
    log.info(f'Infer on {model.getDevice()}')
    infer_request = model.getInferRequest()
    infer_request.infer(inputs)
    res = dict()

    fake_num = 0
    for output in infer_request.results:
        if len(output.get_names()) == 0:
            log.info(f"There is no name for output: {output}, please confirm it")
            output_name = 'fake_output_name_' + str(fake_num)
            fake_num += 1
        else:
            output_name = output.any_name
        res[output_name] = get_infer_results(infer_request, output)
    return res

### 
# Tensor Base
###

def get_precision(precision):
    if precision == 'f32':
        return np.float32
    if precision == 'f16':
        return np.float16
    if precision == 'u8':
        return np.uint8
    if precision == 'i8':
        return np.int8
    if precision == 'i32':
        return np.int32
    if precision == 'i64':
        return np.int64
    return Exception(f"Unsupported precision {precision}")

@error_handling('processing input precision')    
def process_input_precision(input_names, precision_list):
    precision_dict = process_dict_or_list_args(precision_list, input_names)

    if len(precision_list) == 1:
        for key in input_names:
            precision_dict[key] = get_precision(precision_list[0])
    else:
        for key, value in precision_dict.items():
            precision_dict[key] = get_precision(value)
    return precision_dict

    
@error_handling('reading binary file')
def read_binary_file(bin_file, model_input, precision):
    binary_file_size = os.path.getsize(bin_file)
    tensor_size = model_input.tensor.size
    model_dtype = get_dtype(model_input.element_type)

    if tensor_size != binary_file_size:
        log.warning(f"File {bin_file} contains {binary_file_size} bytes but IR model expects {tensor_size}. "
                     f"You set precision to {precision} with shape{model_input.tensor.shape}, but IR's actual type is {model_dtype}")
        return np.reshape(np.fromfile(bin_file, precision), list(model_input.shape)).astype(model_dtype)
    
    return np.reshape(np.fromfile(bin_file, model_dtype), list(model_input.shape))


@error_handling("input processing")
def input_processing(model_inputs, input_path, precision_list):
    log.info("====================== Processing Inputs ======================")
    inputs = [input for input in input_path]
    input_names = [input.any_name for input in model_inputs]
    log.info(f'input_names: {input_names}')
    input_data = {}

    if len(inputs) != len(model_inputs):
        raise Exception('Please provided the matched input')
    
    precisions = process_input_precision(input_names, precision_list)

    for i in range(min(len(inputs), len(model_inputs))):
        tensor_name = os.path.splitext(os.path.basename(inputs[i]))[0]
        # splited = inputs[i].rsplit(':', maxsplit=1)
        # print(inputs)
        # if len(splited) == 0:
        #     raise Exception(f"Can't parse {'input_file'} input parameter!")
        # tensor_name = None
        # if len(splited) == 1:
        #     tensor_name = input_names[i]
        # else:
        #     tensor_name = splited.pop(0)
        if tensor_name not in input_names:
            raise Exception(f"Input with name {tensor_name} doesn't exist in the model!")

        path = Path(inputs[i])
        if path.exists() and path.is_file():
            log.info(f"::: Reading input \"{tensor_name}\" from {str(path)}")
            current_input = model_inputs[input_names.index(tensor_name)]
            input_data[tensor_name] = read_binary_file(path, current_input, precisions[tensor_name])
            # print(input_data[tensor_name])
            # input_data[tensor_name] = read_binary_file(path, current_input, np.float32)
    return input_data

