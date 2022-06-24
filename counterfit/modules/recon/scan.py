import requests

"""
    server: envoy
    content-length: 0
    x-envoy-upstream-service-time: 1
    x-request-id: b24026d3-241d-45d3-8dc2-0371e2d67120
    ms-azureml-model-error-reason: model_error
    ms-azureml-model-error-statuscode: 404

    Accept: Accept
    X-Amzn-SageMaker-Custom-Attributes: CustomAttributes
    X-Amzn-SageMaker-Target-Model: TargetModel
    X-Amzn-SageMaker-Target-Variant: TargetVariant
    X-Amzn-SageMaker-Target-Container-Hostname: TargetContainerHostname
    X-Amzn-SageMaker-Inference-Id: InferenceId


    x-jupyterhub-version: 1.5.0

    {["TorchServe"] = {"GET","/api-description"},
		 ["TensorFlow Serving"] = {"POST", "/v1/models/*:predict", "{\"instances\": [1.0,5.0]}"},
		 ["MlFlow"] = {"POST", "/invocations", "{\"columns\":[\"alcohol\", \"chlorides\", \"citric acid\", \"density\", \"fixed acidity\", \"free sulfur dioxide\", \"pH\", \"residual sugar\", \"sulphates\", \"total sulfur dioxide\", \"volatile acidity\"],\"data\":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}",{ ["Content-Type"] = "application", ["format"] = "pandas-split"}}
	 	}
"""


class EndpointScanner:
    pass