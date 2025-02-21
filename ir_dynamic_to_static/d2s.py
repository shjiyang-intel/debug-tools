import openvino as ov 
from argparse import ArgumentParser
import re 
from pathlib import Path

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
    args.add_argument("-o", "--output", help="Required. Path to output folder, static IR will be renamed to \{model_name\}_static.xml", required=True, type=str)
    args.add_argument("--inputs", help="Required. Path to a .xml file.", required=True, type=str)

    parsed_args = parser.parse_args()
    return parsed_args

if __name__ == '__main__':
    args = build_argparser()
    inputs = parse_inputs(args.inputs)
    model_name = Path(args.model).stem
    core = ov.Core()
    model = core.read_model(args.model)
    
    model.reshape(inputs)
    
    ov.save_model(model, Path(args.output) / model_name / '_static.xml')
