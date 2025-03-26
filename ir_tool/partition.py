from argparse import ArgumentParser, SUPPRESS
import logging
import sys
import os
import numpy as np

from openvino.runtime import Core, Model, serialize, Output
from openvino.runtime import opset9 as opset
import  openvino.runtime.op as op
from openvino.runtime.op import Parameter, Result, Constant
from openvino.runtime.utils import replace_node


# Set up logging to output to console
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO,
    format='[INFO] %(message)s'    
)

log = logging.getLogger(__name__)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    
    args.add_argument("-m", "--model", help="Required. Path to a .xml file.", required=True, type=str)
    
    args.add_argument("-i", "--inputs", help="Optional. The new inputs of the subgraph. eg: -i Add_0 Add1 ", nargs='+', required=False, type=str, default=None)
    args.add_argument("-o", "--outputs", help="Optional. The new outputs of the subgraph. eg: -o Add_0 Add1 ", nargs='+', required=False, type=str, default=None)

    args.add_argument("-ip", "--input_precision", help="precision of new inputs, defualt is FP32", nargs='+', default="FP32", type=str, required=False)
    args.add_argument("-op", "--output_precision", help="precision of new outputs, default is FP32", nargs='+', default="FP32", type=str, required=False)

    args.add_argument("-n", "--model_name", help="The result partion model's name, defual equal to the new in/output name.", type=str, required=False)
    args.add_argument("-out_dir", "--output_dir", help="Output Saved folder", type=str, required=True)
    
    args.add_argument("-s", "--separation", help="Separate model to multiple parts using the provided node as breakpoints, this function is not MATURE!!!", nargs='+', required=False, type=str, default=None)

    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    parsed_args = parser.parse_args()
    return parsed_args

def get_new_params(model, name_to_node_mapping, new_inputs = None):
    """
    process new inputs
    """
    params = []
    if new_inputs is None:
        return [tensor.node for tensor in model.inputs]
    
    for input_name in new_inputs:
        if input_name not in name_to_node_mapping.keys():
            raise ValueError("The new input {} is not in the model".format(input_name))
        
        input_node = name_to_node_mapping[input_name]
        if input_name in [tensor.node.get_friendly_name() for tensor in model.inputs]: 
            params.append(input_node) 
            continue 
        
        # log.info(f"New Input Node name : {input_name}, Node : {input_node}")

        # new_param = opset.parameter(shape=input_node.shape, dtype=input_node.get_element_type(), name=f'{input_name}')
        new_param = opset.parameter(shape=input_node.shape, dtype=input_node.get_element_type(), name='time_proj')
        replace_node(input_node, new_param)

        params.append(new_param) 
    return params

def get_new_results(model, name_to_node_mapping, new_outputs = None):
    results = []
    if new_outputs is None:
        return model.get_results() 
    
    for output_name in new_outputs:
        if output_name not in name_to_node_mapping.keys():
            raise ValueError("The new output {} is not in the model".format(output_name)) 
        
        output_node = name_to_node_mapping[output_name] 
        # log.info(f'New Output Node name: {output_name}, Node : {output_node}')
        for node_out in output_node.outputs(): 
            new_result = opset.result(node_out, name=f'{output_name}') 
            results.append(new_result) 
    if not results: 
        results = model.get_results() 
    return results

def get_model_name(args):
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = ''

        if args.inputs:
            model_name += '_In'
            for input_name in args.inputs:
                model_name += '_[' + input_name.replace('/', '-') + ']'
        
        if args.outputs:
            model_name += '_Out'
            for out_name in args.outputs:
                model_name += '_[' + out_name.replace('/', '-') + ']'
    
    return model_name

def get_name_param_from_mapping(mapping):
    names = []
    param_nodes = []
    for name, node in mapping.items():
        if isinstance(node, Parameter):
            names.append(name)
            param_nodes.append(node)
    return names, param_nodes

def set_node_to_param(name, node):
    param_mapping = dict()
    for i in range(node.get_output_size()):
        new_param = opset.parameter(shape=node.output(i).shape, dtype=node.output(i).get_element_type(), name=f'{name}')
        node.output(i).replace(new_param.output(0))
        param_mapping.update({new_param.get_friendly_name(): new_param})
    return param_mapping

def remove_unused_parm(model):
    op_names = [op.get_friendly_name() for op in model.get_ops() if not isinstance(op, Parameter)]
    for p in model.parameters:
        del_flag = True
        for child in list(p.output(0).get_target_inputs()):
            child_node_name = child.get_node().get_friendly_name()
            if child_node_name in op_names:
                del_flag = False
        
        if del_flag:
            # print('removing unused parameter name: {0}, node: {1}'.format(p.get_friendly_name(), p))
            model.remove_parameter(p)

    return model

def maintain_mapping(model, all_mapping, processed_mapping, remain_nodes):
    cur_processed_mapping =  {op.get_friendly_name(): op for op in model.get_ops()}
    processed_mapping.update(cur_processed_mapping)
    unprocessed_mapping = dict()
    # new_params_mapping should contain new parameters and original remaining unprocessed params
    new_params_mapping = dict()

    for name, node in all_mapping.items():
        # currently, remain_nodes should only have one item, so it is not a list but a name(str) only
        if name not in processed_mapping or name == remain_nodes:
            unprocessed_mapping.update({name: node})
            
            if isinstance(node, Parameter):
                new_params_mapping.update({name: node})
        else:
            for oi in range(node.get_output_size()):
                for child in list(node.output(oi).get_target_inputs()):
                    child_node_name = child.get_node().get_friendly_name()
                    
                    if child_node_name not in processed_mapping:
                        # TODO: should push const into unprocessed mapping or all mapping?
                        if not isinstance(node, Constant):
                            new_param = opset.parameter(shape=node.output(oi).shape, dtype=node.output(oi).get_element_type(), name=f'{name}_o{oi}_child_{child_node_name}')
                            # print('new param: ', new_param)
                            node.output(oi).replace(new_param.output(0))
                            new_params_mapping.update({new_param.get_friendly_name(): new_param})
                            # push new params as new node to the all mapping
                            unprocessed_mapping.update({new_param.get_friendly_name(): new_param})
    
    return processed_mapping, unprocessed_mapping, new_params_mapping

def save_model(model, model_name, args):
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    
    path = os.path.join(args.output_dir, model_name + '.xml')
    serialize(model, path, os.path.join(args.output_dir, model_name + '.bin'))

    log.info(f'New Model parameters: {model.parameters}')
    log.info(f'New Model results: {model.results}')
    log.info("Partition model saved to '{}'".format(path))
    print("===============================================")

def main():
    args = build_argparser()
    core = Core()

    model = core.read_model(args.model)

    name_to_node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
    processed_mapping = dict()
    orig_res = model.outputs

    if args.separation:
        # TODO: support multi separation points in one subGraph
        cut_points = args.separation
        new_params = model.parameters

        for idx, point in enumerate(cut_points):
            point_node = name_to_node_mapping[point]
            new_results = get_new_results(model, name_to_node_mapping, [point])

            new_model = Model(new_results, new_params)
            new_model = remove_unused_parm(new_model)
            save_model(new_model, str(idx) + '_' + point.replace('/', '-'), args)

            # maintain processed mapping and unprocessed mapping
            # but need to add back the current point into unprocessed mapping since it will be next subgraph's param
            processed_mapping, name_to_node_mapping, params_mapping = maintain_mapping(new_model, name_to_node_mapping, processed_mapping, point)
            
            new_params = []
            new_params.extend(list(params_mapping.values()))
            last_point = set_node_to_param(point, point_node)
            new_params.extend(list(last_point.values()))

        # process the last point -> result 
        last_model = Model(orig_res, new_params)
        last_model = remove_unused_parm(last_model)
        save_model(last_model, str(idx + 1) + '_lastPart', args)
        return 0

    new_params = []
    if args.inputs is not None:
        new_params = get_new_params(model, name_to_node_mapping, args.inputs)
    new_params.extend(model.parameters)
    new_results = get_new_results(model, name_to_node_mapping, args.outputs)

    new_model = Model(new_results, new_params)
    new_model = remove_unused_parm(new_model)
    model_name = get_model_name(args)

    save_model(new_model, model_name, args)



if __name__ == '__main__':
    sys.exit(main() or 0)