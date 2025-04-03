import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)

import math
def get_positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)
    return pos_enc


# Dont have to write code in here, these are just abstract methods
class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        #raise NotImplementedError()
         
        """
        Transformer layer
        - https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
        - torch has premade encoder and decoder only layers (this corresponds to as seen in the transformer paper of the encoder and decoder blocks)
        - we're recommended to use decoder only but use torch's TransformerEncoderLayer which follows the encoder architecture in paper 
        so the MultiHeadAttention self attention then feedforward (linear, relu, linear) with the layernorms and residual connections
        - Reasons why doing using torchencoderlayer:
            - TransformerEncoderLayer includes self-attention, and when used with a causal mask, it behaves like a decoder.
            - both TransformerEncoderLayer and TransformerDecoderLayer works. 
                - But TransformerEncoderLayer:
                    - is more memory efficient (doesn't require encoder output), since we want decoder-only model
                    - it's simpler, doesn't support cross attention - which we don't need
        """
        
        # params corresponding to transformer layer
        num_layers=6   # 3 , 512
        num_heads=8
        dim_feedforward=2048 # default was 2048 which is too much, this corresponds to the linear(d_model, dim_feedforward), relu(), linear(dim_feedforward, d_model)
        #dropout=0.1 # 0.1 used in transformer paper
        # norm_first=True: in deeplearning lecture, doing norm first before attention was better but not done in orig transformer paper
        #activation="relu" # can change to use gelu
        # we set batch_first=True bc our input's first dimension is batch
        
        """
        Fine tuning Transformer params
        - num_layers=4, dim_feedforward=2048 was bad / slowly improved
        - Best so far: num_layyers=2, dim_feedforward=512
        - Best so far: num_layyers=1, dim_feedforward=512
        """
        
        encoder_layer=torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=num_heads, dim_feedforward=dim_feedforward, norm_first=True, batch_first=True)

        # Model LAYERS
        self.embed=torch.nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_latent)
        self.transformer_encoder=torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers) # NOTE: have to place TransformerEncoderLayer within TransformerEncoder()
        
        #self.relu=torch.nn.ReLU()
        #self.layer_norm=torch.nn.LayerNorm(n_tokens)
        
        self.linear=torch.nn.Linear(d_latent, n_tokens)
        #self.softmax=torch.nn.Softmax(dim=-1) # DONT use softmax bc want logits not probabilities and logits comes from output of the linear layer so linear is final layer
        
        # dimension params for embedding, transformer, and linear are correct
        
        """
        Softmax Example:
            m = torch.nn.Softmax(dim=-1)
            input = torch.randn(2, 3, 3)
            output = m(input)
            print(output)
            
            tensor([[[0.5209, 0.4005, 0.0786],
             [0.2409, 0.1279, 0.6313],
             [0.3267, 0.0560, 0.6173]],

            [[0.1546, 0.7600, 0.0854],
             [0.4108, 0.5040, 0.0852],
             [0.4260, 0.4499, 0.1240]]])
             -> can see with dim=-1, we get intended effect of the last dimension (across columns) sum to 1
        """
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        #raise NotImplementedError()
        
        
        # CANNOT train autoregressive model on CPU, way too slow, use collab gpu
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("CUDA not available, using CPU")
            device = torch.device("cpu")
            
        
        """
        Input
        # NOTE: input is size (Batch x height x width) and are the compressed tokenized version of images generated from output of our BSQ model
        # -> output of BSQ model is the output of last layer being the decoder from autoencoder model being UnpatchifyLinear(): (B, H * patch_size, W * patch_size, 3)
        # -> note UnpatchifyLinear() returns the original image so (Batch x Original Height x Original Width x 3 Color Channels)
        # -> but for some reason input is (Batch x Height x Width)?
        # -> if print out values of input x, can see it is integers this is bc they are the tokenized version of our images done by our bsq model
        """        
        #print(x.shape) #torch.Size([64, 20, 30])
        
        """
        Mask
        - mask we're using (already made mask): 
        - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer.generate_square_subsequent_mask
        - creates a square mask matrix of size (sz x sz) where sz is the sequence length or number of tokens. The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0)
        - -inf marks invalid locations so this mask prevents from looking at future locations (ith row is current location and jth column are the possible locations)
        example output of sz=5:
            torch.Size([5, 5]) 
            tensor([[0., -inf, -inf, -inf, -inf],
                    [0., 0., -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., 0.]])
                    
        Shift over by 1: 
        - notice the above tensor is not fully correct bc we dont want the current location to attend to itself 
         (at location i, should not be able to attend j=i, rows are i and columns are j)
        - use ConstantPad1D and slicing
            sz=5
            m = torch.nn.Transformer.generate_square_subsequent_mask(sz)
            print(m)
            pad = torch.nn.ConstantPad1d((0, 1), float('-inf')) # 2nd param corresponds to what value to add, tuple in first param corresponds to how many of the value to add to the left and right
            print(pad(m)[:, 1:])
            
            tensor([[0., -inf, -inf, -inf, -inf],
                    [0., 0., -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., -inf],
                    [0., 0., 0., 0., 0.]])
            tensor([[-inf, -inf, -inf, -inf, -inf],    -> "Shifting done by adding a -inf to right side then only including columns starting from index 1"
                    [0., -inf, -inf, -inf, -inf],
                    [0., 0., -inf, -inf, -inf],
                    [0., 0., 0., -inf, -inf],
                    [0., 0., 0., 0., -inf]])
        """
        # pad=torch.nn.ConstantPad1d((1, 0), float(0))
        sequence_len=x.shape[1] * x.shape[2] # height * width is the number of tokens or sequenceLength bc this is the total num of pixels in an image
        mask=torch.nn.Transformer.generate_square_subsequent_mask(sz=sequence_len).to(device) 
        #pad=torch.nn.ConstantPad1d((0, 1), float('-inf'))
        #mask=(pad(mask)[:, 1:]).to(device) # interacting with model which is on gpu so place on gpu
        #print(mask)
        
        """
        Flatten
        - Turn tokenized image of size height x width to a flattened vector of tokens
        - The shape of the input before flattening is shape of x torch.Size([64, 20, 30]) -> 64 corresponds to batch_size
        - The shape of the input after flattening is shape of x torch.Size([64, 600])
        """
        x=torch.flatten(x, start_dim=1) # start_dim=1 bc dont want to flatten the batch dimension so start flattening at dim 1
        #print(x.shape) # torch.Size([64, 600])
        
        """
        Pass through model
        """
        #print(x)
        
        x=self.embed(x) 
        #print(x.shape) # torch.Size([64, 600, 128]) # (batch x sequence_len x embed_dim) 
        # -> sequence_len is the token dimension each represented by a 128 size vector
        
        """
        Shift on output of embedding
        Ex:
        shifted_x = torch.roll(input, shifts=1, dims=-1)  # Shift left by 1
        print(shifted_x)
        shifted_x[:,:,0]=0
        print(shifted_x)
        
        tensor([[[ 1.9148, -0.1426, -0.3177],
                [-0.5510,  0.2510, -0.0982],
                [-1.3463,  1.8922, -0.8155]],

                [[-0.4588, -0.6706,  0.4162],
                [ 0.0731, -1.3510,  1.8573],
                [-0.9159, -1.6018,  0.1288]]])
        tensor([[[ 0.0000, -0.1426, -0.3177],
                [ 0.0000,  0.2510, -0.0982],
                [ 0.0000,  1.8922, -0.8155]],

                [[ 0.0000, -0.6706,  0.4162],
                [ 0.0000, -1.3510,  1.8573],
                [ 0.0000, -1.6018,  0.1288]]])
        -> THIS IS INCORRECT SHIFTING
        """
        # x=torch.roll(x, shifts=1, dims=-1) # dims=-1
        # x[:,:,0]=0 
        
        # Correct Shifting
        x=torch.roll(x, shifts=1, dims=1) # shfit along token dimension so dim 1 (batch x sequence_len x embded_dim)
        x[:,0,:]=0 # padded token to first token/row of x
        # do not make padded token float("-inf") bc loss becomes nan -> does not correspond to mask's -inf meaning (doesnt read it as dont visit that spot and instead will perform computation's with it so def do not use -inf)
        """
        Correct Shifting Toy Example:
        - perform shifting on output of embedding layer so after it
        - after embedding layer we have (batch x seuqenceLength_OR_numOfTokens x embed_dim)
        - so each image represented by tokens and each token is represented by a vector of size embed_dim-> from embedding layer
        - we want to shift the tokens by 1 so model doesnt cheat by looking at current locations's token so shift by 1 along the token dimension (dim=1)
        - then place zeros or some padded token in the first token's vector just as placeholder so x[:,0,:]=0 
        
        import torch
        input=torch.randn(2,3,3)
        print(input, "\n")
        shifted_x = torch.roll(input, shifts=1, dims=1) 
        print(shifted_x, "\n")
        shifted_x[:,0,:]=0
        print(shifted_x)
        
        tensor([[[ 1.4988, -0.6574, -0.1588],
                [-1.8557, -0.0216,  0.3114],
                [-1.2772,  2.6398, -1.1519]],

                [[ 0.0350,  1.8701,  0.0237],
                [ 0.8347, -0.6082,  1.1372],
                [ 1.6094,  0.3436, -0.4911]]]) 

        tensor([[[-1.2772,  2.6398, -1.1519],
                [ 1.4988, -0.6574, -0.1588],
                [-1.8557, -0.0216,  0.3114]],

                [[ 1.6094,  0.3436, -0.4911],
                [ 0.0350,  1.8701,  0.0237],
                [ 0.8347, -0.6082,  1.1372]]]) 

        tensor([[[ 0.0000,  0.0000,  0.0000],
                [ 1.4988, -0.6574, -0.1588],
                [-1.8557, -0.0216,  0.3114]],

                [[ 0.0000,  0.0000,  0.0000],
                [ 0.0350,  1.8701,  0.0237],
                [ 0.8347, -0.6082,  1.1372]]])
        - can see first token is filled with zeros (as like a padding) and the tokens shifted down 1
        """
 
        positional_encoding = get_positional_encoding(seq_len=sequence_len, d_model=x.shape[2])
        x = x + positional_encoding.to(device)
        
        x=self.transformer_encoder(x, mask=mask, is_causal=True) # set is_causal to true bc the mask we're using is causal meaning mask that prevents from looking at future tokens
        #print(x.shape) # torch.Size([64, 600, 128])
        
        #x=self.relu(x)
         
        x=self.linear(x)
        #print(x.shape) # torch.Size([64, 600, 1024])
        
       
        #x=self.layer_norm(x)
        
        # DO NOT USE SOFTMAX bc output is expected to be logits not probabilities and logits come from the above linear layer so linear layer should be the last layer
        #x=self.softmax(x) 
        #print(x.shape) # torch.Size([64, 600, 1024])
        #print(x.sum(dim=-1))
        return x, {}
        
    # This is for part 4 (the generate part of the assignment)
    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        #raise NotImplementedError()
        # img = torch.zeros((B,h,w), dtype=torch.long).to(device) # do not pass in -1 or will get cuda error when shifting in forward()
        # img_shape=img.shape
        # index=0
        # for height_i in range(img_shape[1]):
        #   for width_j in range(img_shape[2]):
        #       # Skip if not to be filled (-1)
        #         # if (img[:,height_i,width_j] != -1).all().item():
        #         #     continue
        #         # pred = forward(img[:,height_i,width_j]) #-> doesnt work, dimension issues
        #         pred, _ = self.forward(img[:,:height_i+1,:]) # need the +1
        #         #pred = self.forward(img)
        #         # if index==0:
        #         #     print(pred.shape)
             
        #         probs = torch.nn.functional.softmax(pred[:, index, :], dim=-1)
        #         #if index==0:
        #         #  print(probs.shape)
        #         #  print(torch.multinomial(probs, num_samples=1).squeeze(dim=-1).shape)
        #         img[:,height_i,width_j] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)
        #         index+=1
        # return img
        
        # img = torch.zeros((B,h,w), dtype=torch.long).to(device)
        # index=0
        # for height_i in range(img.shape[1]):
        #     for width_j in range(img.shape[2]):
        #         pred, _ = self.forward(img[:,:height_i+1,:width_j+1])  # need the +1 (b/c say :2 would acquire only indexes 0,1)
        #         print(pred.shape)
        #         print(height_i, width_j, index)
        #         # print(pred)
        #         # if index==0:
        #         #     print(pred.shape)
        #         probs = torch.nn.functional.softmax(pred[:, index, :], dim=-1)
        #         # if index==0:
        #         #     print(probs.shape)
        #         #     print(torch.multinomial(probs, num_samples=1).squeeze(dim=-1).shape)
        #         img[:,height_i,width_j] = torch.multinomial(probs, num_samples=1).squeeze(dim=-1) # .squeeze bc multinomial produces an extra dimension
        #         index+=1
        # return img
        
        """
        Ideal Generation implementation
        - initialize tensor of zeros of expected image generation shape (batch, h, w)
        - loop through pixels h,w
            - call autoreg model's forward() but pass in only UP TO the ith, jth pixel then auto reg produces 
            logits of (batch, h*w, n_tokens), so every row (dimension h*w) is the set of logits of all possible 
            tokens for that ith, jth pixel
            - turn the logits for the ith, jth pixel into probabilities using softmax only on the pixel 
            being looked at (here represented by index variable -> ith jth pixel) and softmax across the n_tokens dimensions. 
            These probabilties represent the distribution of possible tokens for that ith jth pixel
            - Now want to sample from the distribution of possible tokens to acquire the ith jth pixel's token
            - Repeat until acquired token for every ith, jth pixel
            
        - NOTE: we pass in tokens autoregressively (we start off with zeros, pass in up to the ith jth pixel, and after every token 
        acquired by sampling from generated probs, that predicted token is passed into the next call to 
        forward bc we pass in up to the ith jth pixel so model is using tokens it has predicted to predict next possible tokens)
        
        - IMPORTANT: Inside of forward() it expects (batch, height, width) and does flattenting already. But if we pass in [:, :height_i+1, :width_j+1],
        this indexing wont work once we go to next iter of height_i (b/c the indexing of :width_j+1 will disregard the first row of pixels we just went through )
        Solution: Flatten input before calling forward and add an extra dummy dimension using .unsqueeze(-1). This will allow auto reg indexing the up to the ith pixel and solves the problem of forward() expecting (batch, height, width). Then, lastly after acquiring sampled tokens, reshape the tensor to be shape (batch, height, width). Also, for future reference, I had a dimensionality mismatch issue bc I was unzqueezing the output of the multinomial so I got rid of it and fixed issue.
        """
        
        img = torch.zeros((B,h,w), dtype=torch.long).flatten(start_dim=1).unsqueeze(-1).to(device)
        # print(img.shape)
        for pixel_i in range(img.shape[1]):
            pred, _ = self.forward(img[:, :pixel_i+1])
            # if pixel_i==0:
            #     print(pred.shape)
            probs = torch.nn.functional.softmax(pred[:, pixel_i, :], dim=-1)
            # if pixel_i==0:
            #     print(probs.shape)
            #     print(img[:,pixel_i].shape)
            #     print(torch.multinomial(probs, num_samples=1).shape)
            img[:,pixel_i] = torch.multinomial(probs, num_samples=1)
        # img = img.reshape(B, h, w)
        # print("Image generated shape: ", img.shape)
        # print("Image generated values:\n", img)
        return img.reshape(B,h,w)


"""
Transformer Notes:
- Attention is basic building block of transformer architecture, it is similar to convolution but instead works with a variable length size input
- Attention is a set operator that learns to reason about the structure of a set of elements
- Purpose of attention: To reason over sets of elements and relate individual elements with one another 
- Attention operator uses a query, set of keys, and set of values
    - Each value has a key
    - Each word is a value which is the meaning of the word
    - Query allows us to exchange information between the keys/values with the query
    - Query will measure distance from itself to all keys (dot product q and k) then scaled by C being # keys, exponentiate it, then normalize it
    - This ends up being the softmax of single query and distance to all keys
    - Alpha i is the resulting weight that will all sum to 1 and are always positive
    - Then compute weighted average (omega o)
    - If key is close by then value will have high weight and otherwise will have weight close to 0 or negative having little effect on weighted sum
    - Attention is most likely written as matrix operator where we dont have 1 query but a set of n queries, will compute distances between all queries 
    and all keys, then normalize, then compute softmax (property of softmax is all will sum to 1) and then will retrieve weighted avg of values v
        - IMPORTANT
        - Notice what is being done:
        - Take dot product of Q query and K keys then divide by sqrt C to scale, then softmax is done to normalize (values between 0 and 1 that sum to 1)
        - Then take dot product with V values ( practically weighting importance of values in V based on results from Q*K)
        
- preprocessing for language models:
    - split sentence into parts: characters, words, tokens
    - then embed each part (translate the part (set of chars) to a represnetable vector of numbers
    - Embedding associates a certain set of feature vectors with a specific input (basically translates the part/token into something the network understands being numbers then attention is fed this to relate the #s to one another)
        - Take sentence and parse it
        - Turn parts into #s through embedding
        - Feed embedding into network that will learn to relate #s to one another
        
- Cross Attention and Self Attention    
    - Cross Attention: Query comes from its own input and keys and values come from the same input -> can think about trying to relate spanish words to english words
    - Self Att: Query, keys, and values come from the same input
    - In cross attention, half input goes to query and other half to keys and values
    - Self attention is used in NLP and cross attention is used in computer vision tasks 

- Multi Head Attention
    - Almost no one uses basic attention operator, multi head attention is what people actually use
    - Problem for wanting to deal with arbitrary sequences: An attention always attends more to itself than anything else
    - Basically what the issues of self attention is trying to say is that the attention head when observing an ith row and comparing to other columns, will pay attention to itself more than other tokens bc the dot product has larger value
    - Now apply a linear layer to each input (query, key, value) (creating weight matrices) which allows us to have arbitrary values for query keys and values and then compute attention
    - This trick allows attentions to attend more to other elements rather itself which allows the attention to look at say arbitrary spots in the image
    - Adding the weights made the attention as expressive as a 1x1 convolution but issue is attention operator uses the same attention matrix so cannot gather information from multiple disjointed places, or if it can, it has to average the information together. 
    - SO solution is multi head attention
        - is concatentaiton of multiple attention layers (heads)
    - work flow of MHA
        - h heads (attention layers)
        - each starts with input key, value, query and performs linear projection (linear layer) then is fed into attention operator (scaled Query * key) * value
        - then concat the heads
        - finally perform single linear projection (layer) on concatenated outputs
    - self attention with weights generalizes a 1x1 conv2D
    - MHA with h heads is more expressive than a convolution with sqrt(h) x sqrt(h) kernel size
    
    - typically use 8 or more heads

- The attention scores, computed based on the query and key dot product, are used to weight the value vectors. 
- Higher attention scores mean that the corresponding values are more important for the output.


- Attention with weights is permutation invariant meaning
- Can take bunch of inputs to attention keep queries same , permute all keys and values in arbitrary order, as long as you permute them the same, the output of the attention is the same
- This is a problem bc now we cant reason to the left and right of a location in a sentence. If we use an image, we cannot reason about any spatial arrangement
- Spatial location matters bc ordered operations relies on spatial location and say reshuffling of parts of pics changes meaning
- Solution: add Positional embedding to input (there are different kinds of positional embeddings)
    - absolute, sinusoidal, learnable, relative, rotary
    
- Transformer architecture is compact and computationally effective way to model set operations or even operations over arbitrarily sized inputs
- At its core uses multi head attention and then uses positional embedding to get rid of permutation invariance 
 and combine with multi layer perceptron MLP Multi Layer Perceptron(linear followed by non linear sets of layers)
- add layer norm and residual connections to avoid vanishing gradients
- Look at lecture 6.6 deep learning for visual of architecture
- Placing normalization before transformer (MHA) is computationally faster and better (not done in original transformer paper)
- Transformer layer=MHA(multi head attention) + MLP (multi layer perceptron) + LN (layer norm) + residual connections

lecture 6.8 deep learning
- Transformer used for auto regressive prediction: predicting one token at a time and each next word is conditioned on the previous predictions (what has seen so far)
- By using masking, we force the transformer to only look at words that came before the current token being predicted
- Vanilla transformer has attention to all inputs so it could easily cheat by looking at the token in the input at the current location and keep predicting that
- To mask you add a mask matrix to the attention operator softmax( (Q*K)/sqrt(C) + MaskMatrix) * V
- Mask matrix sets certain elements in the softmax to 0 then attention only goes backwards (only look at tokens before the current token being predicted)
- TYPES OF TRANSFORMERS:
    - Decoder only: masked auto regressive prediction
    - Encoder only: no prediction, just understanding

**** Example 6.9 Deep Learning Good Example. Look at Understnading doc and code
**** Also look at my annotated Transformer Paper

Masking Done in Transformer Paper:
- Masking here prevents the ith token/position being predicted from looking at future (subsequent) positions. 
- Also the output embeddings being shifted by 1 prevents the ith position from cheating by looking at it's ith position 
- (only look at positions less than i)
- negative infinity corresponds to illegal locations/connections 
-> can view a matrix mask as connections. Rows are the ith location and columns are jth location. So if an entry is -inf, then at location i, you cannot attend to location j

- Transformer paper uses learned embedding


"""