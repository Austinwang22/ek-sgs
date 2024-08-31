from abc import ABC, abstractmethod
from functools import partial
import yaml
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from motionblur.motionblur import Kernel

from utils.img_utils import Blurkernel, fft2c_new

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class BaseOperator(ABC):
    def __init__(self, sigma_noise=0.0):
        self.sigma_noise = sigma_noise

    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def __call__(self, data, **kwargs):
        # calculate A(x)
        out = self.forward(data, **kwargs)
        # add noise
        return out + self.sigma_noise * torch.randn_like(out)

    def loglikelihood(self, measurement, data, exact=True):
        '''
        Caculate the log-likelihood
        Args:
            - measurements (torch.tensor): 
        '''
        y_diff = measurement - data
        if self.sigma_noise > 0.0 and exact:
            return -0.5 * torch.sum(y_diff.flatten(start_dim=1) ** 2, dim=1) / (self.sigma_noise ** 2)
        else:
            return -0.5 * torch.sum(y_diff.flatten(start_dim=1) ** 2, dim=1)


@register_operator(name='noise')
class DenoiseOperator(BaseOperator):
    def __init__(self, sigma_noise=0.0):
        super(DenoiseOperator, self).__init__(sigma_noise)
    
    def forward(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(BaseOperator):
    def __init__(self, in_resolution: int, scale_factor: int, sigma_noise=0.0):
        super(SuperResolutionOperator, self).__init__(sigma_noise)
        self.out_resolution = in_resolution // scale_factor

    def forward(self, data, # data: (batch_size, channel, height, width)
                **kwargs):
        down_sampled = TF.resize(data, size=self.out_resolution, interpolation=InterpolationMode.BICUBIC, antialias=True)
        return down_sampled
    

@register_operator(name='motion_blur')
class MotionBlurOperator(BaseOperator):
    def __init__(self, kernel_size, intensity, device, sigma_noise=0.0):
        super(MotionBlurOperator, self).__init__(sigma_noise)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        return self.conv(data)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(BaseOperator):
    def __init__(self, kernel_size, intensity, device, sigma_noise=0.0):
        super(GaussialBlurOperator, self).__init__(sigma_noise)
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device).to(torch.float64)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float64))

    def forward(self, data, **kwargs):
        return self.conv(data)


@register_operator(name='inpainting')
class InpaintingOperator(BaseOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, mask, sigma_noise):
        super(InpaintingOperator, self).__init__(sigma_noise)
        self.mask = mask
    
    def forward(self, data, **kwargs):
        if self.mask.device != data.device:
            self.mask = self.mask.to(data.device)
        return data * self.mask
    

# ----------------- Nonlinear Operators ----------------- #
@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(BaseOperator):
    def __init__(self, oversample=0.0, resolution=64, sigma_noise=0.0):
        super(PhaseRetrievalOperator, self).__init__(sigma_noise)
        self.pad = int((oversample / 8.0) * resolution)
        
    def forward(self, data, **kwargs):
        x = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        if not torch.is_complex(x):
            x = x.type(torch.complex64)
        fft2_m = torch.view_as_complex(fft2c_new(torch.view_as_real(x)))
        amplitude = fft2_m.abs()
        return amplitude.to(torch.float64)


@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(BaseOperator):
    def __init__(self, opt_yml_path, device, sigma_noise=0.0):
        super(NonlinearBlurOperator, self).__init__(sigma_noise)
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        print(model_path)
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred
    
@register_operator(name='high_dynamic_range')
class HighDynamicRange(BaseOperator):
    def __init__(self, device='cuda', scale=2, sigma=0.05):
        super().__init__(sigma)
        self.device = device
        self.scale = scale

    def forward(self, data, **kwargs):
        return torch.clip((data * self.scale), -1, 1)