import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from numpy import ndarray
import numpy
import torch
from torch import Tensor

OpticalFlowData = tuple[ndarray, ndarray]


class OpticalFlowGenerator:
    """Optical Flow Generator generates optical flow from two consecutive frames using selected model"""

    def __init__(self,
                 model_type: str,
                 model_pretrained: str,
                 bound: int,
                 ) -> None:

        print(f"Initializing optical flow model ({model_type} -> {model_pretrained})...")

        self.__model_type: str = model_type
        self.__model_pretrained = model_pretrained

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        # check device capability
        # self.__device: torch.device = torch.device("cpu")
        self.__device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.__half_capable: bool = self.__device.type != "cpu"

        # load model
        self.__model: ptlflow.BaseModel = ptlflow.get_model(model_type, model_pretrained)

        # set model precision
        # self.__model = self.__model.half() if self.__half_capable else self.__model.float()

        # set model to eval mode
        self.__model.eval()

        # boolean for traced model
        self.__is_traced: bool = False

        # bound param
        self.__bound: int = bound
        self.__inverse_rgb_bound_2x: float = 255 / (self.__bound * 2)
        self.__inverse_rgb_bound_2x_x_bound: float = self.__inverse_rgb_bound_2x * self.__bound

        print(f"Optical flow model ({self.__model_type} -> {self.__model_pretrained}) initialized")

    def __first_input(self, model_input_1: ndarray, model_input_2: ndarray) -> None:

        print(f"Warming up optical flow model ({self.__model_type} -> {self.__model_pretrained})...")

        # initialize ioadapter
        self.__ioadapter = IOAdapter(
            self.__model, model_input_1.shape[:2]
        )

        # # trace model on first input for adaptive tracing
        # with torch.no_grad():
        #     print("JIT tracing masker model (yolov7-mask)...")
        #     self.__model = torch.jit.trace(
        #         self.__model,
        #         rand_input,
        #         check_inputs=[
        #             model_input,
        #             zero_input,
        #             rand_input,
        #             rand_2_input
        #         ]
        #     )

        #     print("Optimizing masker model (yolov7-mask)...")
        #     self.__model = torch.jit.optimize_for_inference(self.__model)

        # del zero_input
        # del rand_input
        # del rand_2_input
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
        self.__is_traced = True
        print(f"Optical flow model ({self.__model_type} -> {self.__model_pretrained}) warmed up")

    def generate_once(self, image_1: ndarray, image_2: ndarray) -> OpticalFlowData:
        """Generate optical flow from two consecutive frames"""

        if not self.__is_traced:
            self.__first_input(image_1, image_2)

        inputs: dict = self.__ioadapter.prepare_inputs([image_1, image_2])

        with torch.no_grad():
            opt_output: dict = self.__model.forward(inputs)  # type:ignore

        predictions: dict = self.__ioadapter.unpad_and_unscale(opt_output)
        flows: Tensor = predictions['flows'][0, 0]

        flows = self.__inverse_rgb_bound_2x * flows + self.__inverse_rgb_bound_2x_x_bound

        flows[flows < 0] = 0
        flows[flows > 255] = 255

        flows = flows

        flow_np: ndarray = flows.detach().cpu().numpy().astype(numpy.uint8)

        return flow_np[0], flow_np[1]
