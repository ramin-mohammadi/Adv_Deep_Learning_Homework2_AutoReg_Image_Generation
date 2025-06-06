# Homework 2 - Auto-regressive image generation

In this homework, we will train an auto-regressive image generation model from scratch on a dataset up supertuxkart images [here](https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip).
The final model is both a generative model, and an image compressor.

The homework consists of five parts (one optional):

1. Train a patch-level auto-encoder `ae.py`
2. Convert your auto-encoder into a quantizer `bsq.py` using Binary Spherical Quantization
3. Convert the training and validation sets into their tokenized form and train an auto-regressive model
4. Generate samples from the auto-regressive model
5. (Optional) Use the auto-regressive model to compress your images.

Familiarize yourself with the starter code, download and unzip the data.

```bash
wget https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip -O supertux_data.zip
unzip supertux_data.zip
```

We provide dataloaders for the image dataset, and tokenized versions of the dataset in `data.py`.
We also provide a full training script including logging in `train.py`.
To train any model you define in `ae.py`, `autoregressive.py`, or `bsq.py` simply call:

```bash
python -m homework.train NameOfTheModel
```

Feel free to optionally provide `--epochs ..` (to train for longer) or `--batch_size` (to manage memory).
The trainer will produce two files/directories:

- `logs/{date}_{NameOfTheModel}` a tensorboard log (call `tensorboard --logdir logs` to see all logs)
- `checkpoints/{date}_{NameOfTheModel}.pth` the actual model trained

Finally, we provide the tokenization script (`tokenize.py`) that converts a set of images into a torch tensor of tokens.

## Grading Criteria

The grading criteria are part-specific and explained below.
There are 5 pts extra credit to implement compression using the auto-regressive model and entropy coding.

## Patch-level auto-encoder (30 pts)

We start by implementing the patch-level auto-encoder.
The goal here is to take an image of size W=150 x H=100 pixels and split it into equal-sized patches of size PxP (anything from P=5 to P=25 will work here).
For each patch, we compute a d-dimensional embedding using a non-linear function.
The result is an image of size w x h x d image of features.
A decoder then maps these features back into the original image space.

Architecturally almost anything will work here. Use this part of the assignment to get used to the training code, and get warmed up.
Even a linear encoder and decoder work here.

Train your network by calling:

```bash
python -m homework.train PatchAutoEncoder
```

## Patch-level Quantizer (30 pts)

In this part, we implement a simplified version of [Binary Spherical Quantization (BSQ)](https://arxiv.org/abs/2406.07548).
BSQ uses a binary bottleneck of size C where each feature is either -1 or 1. This binary code directly corresponds to an integer token. See `BSQ._code_to_index` and `BSQ._index_to_code` for functions that map a binary vector to an integer feature.
We provide a differentiable binarization function `diff_sign` in `bsq.py`. It allows for a -1 / 1 quantization and uses a straight-through gradient estimator.
Using this function directly in a binary bottleneck likely will not work (see BSQ paper for baselines, or feel free to train it yourself).
However, simply normalizing the inputs to the binarization using an L2-norm will lead to a fairly easy-to-train and efficient quantizer.
Finally, BSQ projects auto-encoder features down to a lower-dimensional bottleneck (also called codebook dimension), then normalizes, quantizes, and projects everything back into the original space.

BSQ optimizes several losses in addition to reconstruction (entropy, commitment, GAN, etc).
For this assignment, you can safely ignore all other losses, a simple differentiable sign and normalization should suffice.

In this part, you should combine your auto-encoder from part 1 with the BSQ quantizer.
It is highly recommended to follow the hyper-parameters set out in the starter code `patch_size=5` and `codebook_bits = 10`, this will make later parts easier.

Train your network by calling:

```bash
python -m homework.train BSQPatchAutoEncoder
```

Make sure to fire up tensorboard `tensorboard --logdir logs` to monitor your training.

A well-trained quantizer will look like this:

![](tensorboard.png)

The above quantizer was trained for an hour in a single GPU.
It introduces some artifacts, due to limited training, and using just an L2 loss, but anything close to this will suffice for the homework.
If your quantizer is blocky, or blurry don't worry, you will still be able to complete the assignment. Training a successful quantizer can take as little as 5min on an entry-level GPU.

With the `BSQPatchAutoEncoder` trained, let's create a token-level dataset for the next part.

```bash
python -m homework.tokenize checkpoints/YOUR_BSQPatchAutoEncoder.pth data/tokenized_train.pth data/train/*.jpg

python -m homework.tokenize checkpoints/YOUR_BSQPatchAutoEncoder.pth data/tokenized_valid.pth data/valid/*.jpg
```

This will create two files `data/tokenized_train.pth` and `data/tokenized_valid.pth` containing the entire training and validation datasets.

If you're curious, check the tokenized checkpoint size:

```
du -hs data/tokenized_train.pth
```

If you follow the hyper-parameters above, it should be around 76Mb (compared to 500Mb for the original JPG dataset).
This is already a great compression result, but we will be able to get it down further.

## Auto-regressive model (30 pts)

Finally, we will train an auto-regressive model.
It takes a batch of the tokenized image as input and produces a distribution over next tokens as output.

Design your `AutoregressiveModel` in `autoregressive.py`.
Many models work here, but a decoder-only transformer might be the easiest.
As with the quantizer above, you will not require a large network to pass this assignment.
We recommend using `torch.nn.TransformerEncoderLayer` (not a typo) with a causal mask `torch.nn.Transformer.generate_square_subsequent_mask`.
For this to work, you should flatten your input image into a sequence first.
You'll need to take care handling the auto-regressive prediction: The output at location (i, j) should not see the input token at location(i, j) which should predict and only see tokens preceding it.
You may use a positional embedding, but this is optional.

Once your network is ready, train it using the next-token cross-entropy loss.

```bash
python -m homework.train AutoregressiveModel
```

Fun fact: The cross-entropy corresponds to the compression rate an arithmetic coding algorithm can obtain using your model.

Your model should be able to reach an average of 4500 bits per image quite easily.
A well-trained model can go as low as 4000 bits per image (500 bytes, an order of magnitude smaller than JPG, although not at the same quality).

## Generation (10 pts)

Finally, produce samples from your generative model by implementing `AutoregressiveModel.generate`.
Since our model is quite small, and the tokenizer is lossy, don't expect great generation results.

Here are samples from a model without positional embedding

![](gen_nopos_1.png) ![](gen_nopos_2.png) ![](gen_nopos_3.png)

As you can see, they mostly capture co-occurrence statistics of patches (within a level).

With a positional embedding, the results look slightly better, but still far from great:

![](gen_pos_1.png) ![](gen_pos_2.png) ![](gen_pos_3.png)

To generate your own samples use

```
python3 -m homework.generation checkpoints/YOUR_TOKENIZER checkpoints/YOUR_AUTOREGRESSIVE_MODEL N_IMAGES OUTPUT_PATH
```

If you trained your model for only a few (even one works) epochs, the generations may look like this:

![](gen_one_1.png) ![](gen_one_2.png) ![](gen_one_3.png)

Getting better generations will require:

1. A better Quantizer (smaller patches or higher bitrates)
2. A much larger transformer
3. Longer training

## Extra credit: Compression (5 pts)

If you want to challenge yourself, try implementing the `Compressor.compress` and `Compressor.decompress` functions in `compress.py`.

## Checkpoints
During training, model checkpoints will be automatically saved in the checkpoints/ directory. The latest trained model will also be saved in the homework/ directory and will be used for grading.

If you wish to submit a specific checkpoint instead of the most recent one, you can manually copy the desired checkpoint from checkpoints/ into homework/, overwriting the existing model file. This ensures that the grader evaluates your preferred checkpoint.

For example, if you trained an AutoregressiveModel on 2025-02-27 and want this checkpoint to be graded, run:

```bash
cp checkpoints/2025-02-27_AutoregressiveModel.pth homework/AutoregressiveModel.pth
```

## Apple Silicon (MPS) and Bitwise Operations Bug

During the implementation of Binary Spherical Quantization (BSQ), we encountered a bug related to bitwise operations on Apple Silicon (MPS) in PyTorch. Specifically, bit shifting operations (<< and >>) do not work correctly when executed on an MPS-enabled device.

This issue is tracked in the PyTorch repository:
[PyTorch Issue #147889](https://github.com/pytorch/pytorch/issues/147889)

To work around this issue, we have implemented a custom bitwise operation using the `diff_sign` function. This function provides a differentiable approximation of the sign function, which is equivalent to the bitwise operation in the reference implementation.

Workaround:
Instead of using bit shifting (x << n), we strongly recommend using exponentiation of 2 (x * (2 ** n)) to ensure compatibility across different hardware.

Example:
Instead of:

```python
index = (binary_code << torch.arange(codebook_bits))
```

Use:

```python
index = (binary_code * (2 ** torch.arange(codebook_bits)))
```

This will avoid computation errors when running the BSQ model on Apple Silicon (e.g., M1, M2, M3).

## Submission

Once you finished the assignment, create a submission bundle using:

```bash
python3 bundle.py homework [YOUR UT ID]

python bundle.py homework rm59549
```

Submit the zip file on Canvas. Please note that the maximum file size our grader accepts is **20MB**. Please keep your solution compact.
Please double-check that your zip file was properly created, by grading it again:

```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through Canvas to grade all your submissions. There is a soft limit of **5** submissions per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of Python and the grader:

- Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
- Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
- Network access is disabled. Please do not try to communicate with the outside world.
- Forking is not allowed!
- `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```


## TA Tips
1/ PAE:
- forward() only has encode() and decode()
- if you have gelu as the last layer of decode, remove it.
- use at least 1 conv and gelu for encode (patchify -> conv -> gelu) and decode (conv -> gelu -> unpatchify). Remember to use chw_to_hwc <-> hwc_to_chw properly.
- when you have additional conv and gelu layers, the PAE performance will reduce, hence setting patch_size = 5 when re-running the PAE model.

2/ BSQ:
- bsqpae.forward() has only encode and decode.
- the bsqpae.encode/decode are used for training, while bsqpae.encode_index/decode_index are used for grading. They have similar structure. If the training loss is good but validation loss (grader) is not good, the problem is that you don’t have similar structure for them.
-  check the paper to understand the model architecture more. The encode in the paper is the pae.encode() or super().encode.
- shouldn’t flatten the input
- shouldn’t use shape conversion (chw <-> hwc).




# My Personal Steps For Running/Training:

- Download Dataset (400MB)
wget https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip -O supertux_data.zip
unzip supertux_data.zip
or just paste link in browser and unzip the file

- follow installation steps above of creating conda env then installing required dependencies by pip installing requirements.txt in the conda env
- had to end up installing tensorboard: pip install -U tensorboardX
- open anaconda prompt and make sure you're in the conda env that was created then run the below python commands
- Important: after unzipping dataset, go into folder and place the "data" folder inside the homework folder (same location as the grader,homework and logs folders)
	- this is bc in data.py, it expects: DATASET_PATH = Path(__file__).parent.parent / "data" -> so 2 parent folders above location of data.py then in a folder called data
	- and inside the data folder from zipped file is the train and val folders
- Want to cd into directory same location as the logs, grader, and homework folders


- Training (run these commands from same location as the logs,homework, and grader)
	- python -m homework.train NameOfTheModel
	- Open Tensorboard during training: -> not necessary
		- pip install tensorboard
		- tensorboard --logdir logs
	
- Sequentially complete
	- Train PatchAutoEncoder
		- python -m homework.train PatchAutoEncoder
	- Train BSQPatchAutoEncoder
		- python -m homework.train BSQPatchAutoEncoder
	- Generate tokenized training and validation images by feeding through BSQPatchAutoEncoder -> if generate a better BSQ, re-run these commands
		- python -m homework.tokenize checkpoints/2025-03-13_05-31-18_BSQPatchAutoEncoder.pth data/tokenized_train.pth data/train/    -> had to get rid of the *.jpg bc command prompt doesnt recognize it (bash/regex command)
		- python -m homework.tokenize checkpoints/2025-03-13_05-31-18_BSQPatchAutoEncoder.pth data/tokenized_valid.pth data/valid/
		- The original train folder is 500MB and my tokenized_train.pth was around 80MB
		- I noticed inside of the train and val folders are .json files and bc we previosuly got rid of the *.jpg, we may be using the .json files in the BSQ model messing with auto reg training
		SOLUTION to the .json files in train and val
		- conda install m2-base    -> allows us to use unix commands like ls in anaconda prompt and *.jpg works, clear, etc. (all the commands we know)
		- python -m homework.tokenize checkpoints/2025-03-13_05-31-18_BSQPatchAutoEncoder.pth data/tokenized_train.pth data/train/*.jpg
		- python -m homework.tokenize checkpoints/2025-03-13_05-31-18_BSQPatchAutoEncoder.pth data/tokenized_valid.pth data/valid/*.jpg
		-> still getting error, i think its from *.jpg not being an exact directory path to a specific .jpg PIL's Image.open() cant interpret *.jpg
		-> so manually remove .json files:
		- rm data/train/*.json
		- rm data/valid/*.json
		- ls data/train/*.json | wc -l    -> count num of .json files in path
		see if only .jpg files left in train and valid (see if they have same count):
		ls data/train/* | wc -l
		ls data/train/*.jpg | wc -l
	- Train AutoRegressive model using the generated tokenized images
		- python -m homework.train AutoregressiveModel
			- NOTE: The cross-entropy (loss used for AutoReg training -> look in train.py) corresponds to the compression rate an arithmetic coding algorithm can obtain using your model.
			- Your model should be able to reach an average of 4500 bits per image quite easily. -> loss

	
- Test using local Grader
	- LOCAL GRADER: python -m grader homework -vv
	- python bundle.py homework rm59549 -> bundle into .zip
	- python -m grader rm59549.zip -> do not use this for local grading
	
	
	
## Results , Remeber a too complicated network (too many layers) increases likely that network dies off 
- PatchAutoEncoder 
	- OLD: 30/30, trained for defualt epochs being 5: Validation loss: 0.0022978780325502157 -> AdamW lr=1e-3 and in ae.py, PatchEncode forward was return self.gelu( self.encode_layer(x) )  and PatchDecode was return self.decode_layer( self.gelu( x ) )
		- this was without the extra conv layers but these are needed for bsq
	- 30/30 -> PAE for BEST BSQPAE train/loss=0.00327, validation/loss=0.00357 -> lr=1e-3, 2 epochs
		- PAE used for best BSQPAE results had overfitting at 5 epochs (loss went up to 0.06) so instead trained for 2 epochs:
		- python -m homework.train PatchAutoEncoder --epochs 2
- BSQPatchAutoEncoder: 
	- OLD: 28/30, 5 epochs training (10 epochs lead to overfitting), Val loss: 0.005297622177749872, lr=1e-3 -> to improve BSQ, added another conv2D 1x1 kernel and ConvTranspose to upscale width height properly and another Gelu, Note increasing channel size did not help
	- BEST BSQPAE RESULTS: 30/30 train/loss=0.00319, validation/loss=0.00373 -> overfitted couldve done 3 or 4 epochs ,5 epochs lr=1e-3, PatchAutoEncoder(PAE): Patchify -> Conv 3x3 kernel padding 1 stride 1 -> gelu -> ConvTransp 3x3 kernel padding 1 stride 1 -> gelu -> Unpatchify
		- interesting that it trained slower and started with a high loss then got so low
		- changes made was kernel 1x1 to 3x3 for extra conv layers and ordering of the gelu and convs
- Trained PAE and BSQPAE on cpu
- learning rate could be adjusted in train.py but left as is being 1e-3 for all 3 models

python -m homework.tokenize homework/BSQPatchAutoEncoder.pth data/tokenized_train.pth data/train/
python -m homework.tokenize homework/BSQPatchAutoEncoder.pth data/tokenized_valid.pth data/valid/

AutoReg error:
RuntimeError: shape '[1, 1, 6000, 6000]' is invalid for input of size 360000
    attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[1, 1, 6000, 6000]' is invalid for input of size 360000
input size of x in AutoReg's forward: 64, 20,30,10
Output of Encoder and decoder of PAE is as expected:
torch.Size([2560, 4, 6, 128])
torch.Size([2560, 100, 150, 3])
-> I FIXED THE ABOVE ERROR, problem was in the BSQPatchAutoEncoder's encoder_index and decoder_index functions, I was getting confused on what to do (using BSQ's encoder or encoder_index and likewise for BSQ's decoder). 
-> Supposed to call PAE encoder then BSQ's encoder_index method then for the decoder_index, do BSQ's decoder_index then PAE decoder

- AutoRegressive
	- input: tokenized images generated from BSQ model (batch x height x width)
	- Flatten input to (batch x (height*width))
	- Embedding Layer
	- Shift output of embedding layer along token dimension to avoid model cheating by looking at current location's token (NOT supposed to shift the mask)
	- + sin cos positional encoding
	- TransformerEncoder
		- torch.nn.Transformer.generate_square_subsequent_mask() as the mask
		- is_causal=True
		- TransformerEncoderLayer`
			- num_layers=6
			- num_heads=8
			- dim_feedforward=2048
			- norm_first=True
			- batch_first=True
	- Linear 
	- DO NOT PERFORM SOFTMAX, expecting logits and not probabilties and logits come from the linear layer so that is the last layer
	- d_latent=128 -> increasing to 256, with above setup, loss was stuck at 6e+3 loss -> so making model too big can be bad
	- 3.8M params -> # params/weights
	
	- main change needed to get model working was performing the shfiting properly and setting the padded token correctly
		- shift along token dimension and set the first token to just 0s
		
	train/loss=1.52e+3, validation/loss=2.8e+3 , 5 epcohs, lr=1e-3, val loss decreased across epochs so could train more
	- Autoregressive prediction loss [ 15 / 15 ]
	- Check autoregressiveness of the model [ 15 / 15 ]
	-> 30/30 for auto reg
	-> Trained on google collab T4 GPU (cannot train on cpu as it is incredibly slow). With above model setup, one epoch is about 7 minutes.
	-> train.py is using lightning module so train/val data and model is moved onto gpu automatically if one is available.
	-> so in autoregressive.py just have to move new variables created interacting with the model onto gpu just being the mask (input x in forward() is the train and val data so it's already on gpu)