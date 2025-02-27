from argparse import ArgumentParser
import re 
from pathlib import Path

import openvino as ov 
import openvino.opset13 as opset
from openvino.runtime.utils import replace_node
from openvino.runtime.op import Parameter, Result, Constant

def parse_inputs(inputs_str):
    pattern = r'(\w+)\[([\d,]+)\]'
    matches = re.findall(pattern,inputs_str)
    inputs = {}
    for name, shape_str in matches:
        shape = list(map(int,shape_str.split(',')))
        inputs[name] = shape
    return inputs

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    
    args.add_argument("-m", "--model", help="Required. Path to a .xml file.", required=True, type=str)
    args.add_argument("--inputs", help="Required. Provide Parameter name and corresponding value e.g. --inputs \"shape[1,8,24,24],b[1,2,3]\".", required=False, type=str)

    parsed_args = parser.parse_args()
    return parsed_args

def convert_param(param_name: str, shape: list):
    global model
    for op in model.get_ordered_ops():
        if isinstance(op, Parameter) and op.get_friendly_name() == param_name:
            new_const = opset.constant(shape,dtype=op.element_type,name=op.get_friendly_name())
            replace_node(op, new_const)
            model.remove_parameter(op)


if __name__ == '__main__':
    args = build_argparser()

    inputs = parse_inputs(args.inputs)
    folder = Path(args.model).parent
    model_name = Path(args.model).stem
    core = ov.Core()
    model = core.read_model(args.model)
    for input in inputs.items():
        convert_param(input[0], input[1])
    ov.save_model(model, folder / (str(model_name) + '_modified.xml'))

