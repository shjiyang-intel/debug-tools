from openvino.runtime import Core, Model, CompiledModel
from io import BytesIO
import logging as log

class ConcreteModel:
    def __init__(self, core: Core, model_path, blob_path, deviceName: str):
        self.__core = core
        self.__device = deviceName

        if (model_path == None and blob_path == None):
            raise Exception("Both runtimeModel and compiledModel are None")

        self.__model_path = model_path
        self.__blob_path = blob_path

        self.__runtime_model = None
        self.__compiled_model = None

        self.setModels()

    def setModels(self):
        if self.__model_path:
            self.__runtime_model = self.__core.read_model(self.__model_path)
        
        if self.__blob_path:
            log.info(f"Read blob from {self.__blob_path} to NPU")
            with open(self.__blob_path, 'rb') as f:
                buf = BytesIO(f.read())
            self.__compiled_model =  self.__core.import_model(buf, 'NPU')

        if self.__runtime_model is None and self.__compiled_model:
            self.__runtime_model = self.__compiled_model.get_runtime_model()

        if self.__compiled_model is None and self.__runtime_model:
            if self.__device == 'NPU':
                log.info("No specific blob provided, need to compile it to NPU")
            self.__compiled_model = self.__core.compile_model(self.__runtime_model, self.__device)

    def getRuntimeModel(self):
        return self.__runtime_model

    def getCompiledModel(self):
        return self.__compiled_model
    
    def getDevice(self):
        return self.__device
    
    def getInferRequest(self):
        return self.__compiled_model.create_infer_request()

    def getModelInfo(self):
        return self.__runtime_model.get_ordered_ops(), self.__runtime_model.inputs, self.__runtime_model.outputs