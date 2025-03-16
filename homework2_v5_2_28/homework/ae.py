import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)

# Pytorch adpoted channel first (c x h x w)
# deep networks that use channel last are faster (h x w x c)
def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))
        # IMPORTANT TO NOTE: only have to make sure dimension is chw for convolutional layers
        # Then we put back into hwc so future operations can be performed quicker

"""
Convolution Notes:
- Convolution is a spatial anchor (sliding window / kernel / patches) linear operator
- But problem is everytime we have a convolution layer, input shrinks. Solution is padding
- If kernel size is kxk, then padding p=(k-1)//2 to maintain original input size after conv layer
- if kernel is say size 3x3, then padding should be 1 -> (3 = 2p + 1 where p = 1)
- You should never have a kernel size that is even, if you want to implement padding
- striding (skipping s pixels for the kernel window) -> will reduce width and height
    - after a conv layer with striding, output size for height and width is divided by the stride
- as you convolute want to increase channels (using out_channels param in Conv2d) 
- maxpooling no longer used because it has become less effective with more complex networks 
    - maxpool original use was to reduce the size of the image but now we just use striding to do this 
    - maxpooling can work as a non linear layer
- To upscale/UpConvolute (increase width and height), use ConvTranspose2d()
    - Up convolution helps increase the resolution of a region.
    - Can be seen as artificially placing zeros into input and running conv on it
    - stride for Conv2d() meant how much we want to downsample, but stride for ConvTranspose2D() means how much we want to upsample
    - padding for ConvTranspose2d() will cut things from the output so padding of 0 will make output larger and padding 1 will cut and shrink the output

- Output size (height x width) after conv layer = ((Input size-Kernel size + 2*Padding)/Stride) +1
- Remember, channel after conv layer determined by out_channels parameter in Conv2d()

- Output Size after convTRANSPOSE layer =(S*(I-1)+K-2P+O)  , S=Stride, I=InputSize, K=KernelSize, P=Padding, O=OutputPadding
"""

"""
Understanding Convolution being done:
- torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

- Patchify Linear
    torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)
        - input to this layer will be (channel x height x width) -> can assume actual size is (batch x c x h x w)
            - which is why in forward() we make sure input to conv is cxhxw
            - Then notice after conv operation in forward(), convert to channel last (hxwxc)
            - This is bc: 
               - Pytorch adpoted channel first (c x h x w) -> so this is expected dim for pytorch
               - But, deep networks that use channel last are FASTER (h x w x c)
        - increase channels from 3 -> latent_dim=128
        - the patch_size=25 params correspond to kernel_size and stride
        - kernel_size corresponds to window size during conv operation (25x25)
        - stride is how many pixels we're skipping as we perform conv operation on kernel window
            - bc stride=25 and kernel_size=25, we're practically only performing conv operation on "patches" of an image
            - ex: think of a 50x50 image (height x width), by having stride=25 and kernel_size=25, we have 4 patches/quadrants of the image that we convolute on
        - outputsize after single one of these conv layers
            - Ex: if input is 3x150x100 (channel x h x w)
            - new_channels=128
            - new_height=((150-25 + 2*0)/25)+1=6
            - new_width=((100-25 + 2*0)/25)+1=4
        - So we end up with a 128x6x4 encoded image that we'll feed through a non linear layer
        - Note DO NOT want to do more than one conv layer bc stride bigger than height and width 
        

- torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

- Unpatchify Linear
    torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)
        - Reverts our output from Patchify Linear to original dimensions (batch x 3 x 150 x 100)
        - ConvTRANSPOSE() allows us to upscale width and height
            - input is (batch x 128 x 6 x 4) (batch x channel x height x width) 
            - channel reduced: 128 -> 3
            - the patch_size params correspond to kernel_size and stride
            - new_height=(S*(I-1)+K-2P+O) = (25*(6-1)+25-(2*0)+0) = 150
            - new_width=(25*(4-1)+25-(2*0)+0) = 100
            - so we end up with original dimension: batch x 3 x 150 x 100
        - again in forward() after perform conv layer, put back into channel last (hxwxc)
"""


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, 3).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, 3) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))
        # IMPORTANT TO NOTE: only have to make sure dimension is chw for convolutional layers
        # Then we put back into hwc so future operations can be performed quicker

"""
 - PatchAutoEncoder inherits this class
 - DONT have to write any code in this Base abc class
 - Its purpose is to define abstract methods (OOP) being required funcitons that have to 
   be defined in whatever class inherits it, which is done in PatchAutoEncoder class
"""
class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x. -> Decode will be as close as possible to original input
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            #raise NotImplementedError()
            self.encode_layer=PatchifyLinear(patch_size, latent_dim) # conv
            self.gelu=torch.nn.GELU() # non linear layer
            #self.conv=torch.nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv=torch.nn.Conv2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.gelu2=torch.nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            #raise NotImplementedError()
            
            # Note, torch conv layer requires channel first format but non linear layers do not require it
            # Output of encode() is a channel last dimension format (batch x height x width x channel) 
            # and in PatchifyLinear() changes to channel first format before performing conv layer

            # implementation for PatchAutoEncoder.pth
            #return self.gelu( self.encode_layer(x) ) 
            
            # -> implementation for BSQ (adding extra conv improved but pick good kernel size -> kernel_size and stride=patch_size was bad)
            #return self.gelu2( chw_to_hwc(self.conv( hwc_to_chw(self.gelu( self.encode_layer(x) ) ) )) ) -> was not better for BSQ
            #return chw_to_hwc(self.conv( hwc_to_chw(self.gelu( self.encode_layer(x) ) ) )) -> MOST RECENT
            # res= self.gelu(chw_to_hwc(self.conv( hwc_to_chw( self.encode_layer(x)  ) ))  )
            # print(res.shape)
            # return res
            return self.gelu( chw_to_hwc( self.conv( hwc_to_chw( self.encode_layer(x) ) ) ) )
            
            # IMPORTANT TO NOTE: only have to make sure dimension is chw for convolutional layers
            # So activation functions (non linear) do NOT require to put dim in chw form
            # so here we return a hwc (channel last) form

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            #raise NotImplementedError()
            self.decode_layer=UnpatchifyLinear(patch_size, latent_dim)
            self.gelu=torch.nn.GELU()
            # self.conv_transpose=torch.nn.ConvTranspose2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv_transpose=torch.nn.ConvTranspose2d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
            self.gelu2=torch.nn.GELU()
   

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            #raise NotImplementedError()
            
            # For decoder, non linear first, then conv
            #return self.decode_layer( self.gelu( x ) )  -> implementation for PatchAutoEncoder.pth
            # res=self.decode_layer( self.gelu( chw_to_hwc(self.conv_transpose( hwc_to_chw(x) ) ) ) )
            # print(res.shape)
            # return res
            #return self.decode_layer( self.gelu2(chw_to_hwc(self.conv_transpose( hwc_to_chw(self.gelu( x )) )) ) ) -> MOST RECENT
            return self.decode_layer( self.gelu( chw_to_hwc( self.conv_transpose( hwc_to_chw(x) ) ) ) )


    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        #raise NotImplementedError()
        self.encoder=self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder=self.PatchDecoder(patch_size, latent_dim, bottleneck)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor: # I will use the PatchEncoder() class instead
        #raise NotImplementedError()
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor: # I will use the PatchDecoder() class instead
        #raise NotImplementedError()
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        #raise NotImplementedError()
        return self.decode( self.encode(x) ), {}