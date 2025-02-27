from argparse import ArgumentParser
import re 
from pathlib import Path

import openvino as ov 
import openvino.opset13 as opset
from openvino.runtime.utils import replace_node
from openvino.runtime.op.util import VariableInfo, Variable

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    
    args.add_argument("-m", "--model", help="Required. Path to a .xml file.", required=True, type=str)
    args.add_argument("-o", "--output", help="Required. Path to output folder, static IR will be renamed to \{model_name\}_static.xml", required=False, type=str)

    parsed_args = parser.parse_args()
    return parsed_args

def convert_gather(model: ov.Model):
    for op in model.get_ordered_ops():
        if op.get_type_name() == "Gather":
            shape = op.output(0).partial_shape
            if len(shape) == 0 and len(op.input(1).get_partial_shape()) == 0:
                # there is case that input's 1D tensor and indice is a value, output will be dynamic with shape: [?], but it's [1] actually
                indice_data = op.input(1).get_source_output().get_node().get_data()
                new_constant_with_shape = opset.constant([indice_data], dtype=ov.Type.i64,name=op.get_name())

                # print(new_constant_with_shape, len(op.input(1).get_partial_shape()))
                replace_node(op,new_constant_with_shape)

def convert_readvalue():
    global model
    for op in model.get_ordered_ops():
        if op.get_type_name() == "ReadValue" and op.output(0).partial_shape.is_dynamic:
            if op.input(0).get_partial_shape().is_dynamic:
                print("ReadValue has dynamic input!")
                exit()
            shape = op.input(0).get_partial_shape()
            
            read_value_attributes = op.get_attributes()
            var_info = VariableInfo()
            print(shape)
            var_info.data_shape = shape
            var_info.data_type = op.get_element_type()  
            var_info.variable_id = read_value_attributes['variable_id']
            variable = Variable(var_info)

            new_read_value = opset.read_value(op.input(0).get_source_output(), variable)
            print(new_read_value.get_attributes())
            replace_node(op, new_read_value)
    return model

if __name__ == '__main__':
    args = build_argparser()
    folder = Path(args.model).parent
    model_name = Path(args.model).stem
    
    core = ov.Core()
    model = core.read_model(args.model)
    # new_model = convert_readvalue()
    for op in model.get_ordered_ops():
        if op.get_type_name() == "ReadValue" and op.output(0).partial_shape.is_dynamic:
            if op.input(0).get_partial_shape().is_dynamic:
                print("ReadValue has dynamic input!")
                exit()
            shape = op.input(0).get_partial_shape()
            
            read_value_attributes = op.get_attributes()
            var_info = VariableInfo()
            print(shape)
            var_info.data_shape = shape
            var_info.data_type = op.get_element_type()  
            var_info.variable_id = read_value_attributes['variable_id']
            variable = Variable(var_info)

            new_read_value = opset.read_value(op.input(0).get_source_output(), variable)
            replace_node(op, new_read_value)
            print(new_read_value.get_attributes())

    ov.save_model(model, folder / (str(model_name) + '_modified.xml'))

