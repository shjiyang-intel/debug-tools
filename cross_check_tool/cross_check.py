import datetime
import logging
import os
import sys
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
from openvino.runtime import Core
from openvino.runtime.utils.types import get_dtype
from utils import *
from compare import metrix_compare

# Set up logging to output to console
logging.basicConfig(
    stream=sys.stdout, 
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'    
)

def main():
    args = build_argparser()
    core = Core()

    ref_combined = ConcreteModel(core, args.model, None, deviceName=args.device)
    if args.npu_outputs and args.blob:
        log.warning("Both npu_outputs and blob are set, please only set one of them, otherwise the NPU model Blob will be ignored")

    npu_combined = ConcreteModel(core, model_path=args.model, blob_path=args.blob, deviceName='NPU')

    model_ops, model_inputs, model_outputs = ref_combined.getModelInfo()

    input_datas = input_processing(model_inputs, args.inputs, args.input_precision)
    ref_res = infer(ref_combined, input_datas)

    if args.npu_outputs:
        log.info("============ Processing NPU output.bin ===================")
        output_names = []
        fake_num = 0
        for output in model_outputs:
            if len(output.get_names()):
                output_names.append(output.any_name)
            else:
                output_names.append('fake_output_name_' + str(fake_num))
                fake_num += 1

        # output_names= [output.any_name if len(output.get_names()) else 'fake_output_name' for output in model_outputs ]
        if len(output_names) != len(args.npu_outputs):
            log.error(f"Model has {len(output_names)} outputs, but you only provided {len(args.npu_outputs)}.")
        npu_outputs = process_dict_or_list_args(args.npu_outputs, output_names)

        npu_res = dict()
        for npu_output_name, tensor in npu_outputs.items():
            # precision = get_dtype(model_outputs[output_names.index(npu_output_name)].element_type)
            precision = np.float32
            shape = model_outputs[output_names.index(npu_output_name)].tensor.shape
            log.info(f"::: Reading output \"{npu_output_name}\" from file {tensor} with dtype {precision}")
            npu_res[npu_output_name] = np.fromfile(tensor, precision).reshape(shape)
    else:
        npu_res = infer(npu_combined, input_datas)

    fake_num = 0
    for output in model_outputs:
        if len(output.get_names()) == 0:
            name = 'fake_output_name_' + str(fake_num)
            fake_num += 1
        else:
            name = output.any_name
        ref_output = ref_res[name]
        npu_output = npu_res[name]
        print('\n')
        log.info(f"Comparing {args.device} VS NPU for {name} with shape {npu_output.shape}: ")
        if (args.print):
            log.info(f'ref_output: {ref_output}')
            log.info(f'npu_output: {npu_output}')
        metrix_compare(npu_output, ref_output)

        if args.mat:
            import matplotlib
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(3)
            axs[0].plot((ref_output.flatten()))
            axs[1].plot((npu_output.flatten()))
            axs[2].plot((ref_output - npu_output).flatten())
            file = name.replace('/', '-')
            plt.subplots_adjust(hspace=1)
            fig.savefig(f'./pngs/{file}.png')

        if args.dump:
            output_folder = os.path.dirname(args.model)
            if args.output_dir:
                output_folder = args.output_dir
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            save_array_to_file(ref_output, folder=output_folder, model_name=get_model_name(args.model), tensor_name=name, device='CPU')
            save_array_to_file(npu_output, folder=output_folder, model_name=get_model_name(args.model), tensor_name=name, device='NPU')

if __name__ == '__main__':
    sys.exit(main() or 0)
