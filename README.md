# Transformers

## Overview
This project implements a Transformer-based Neural Machine Translation (NMT) model for translating sentences between two languages. The model leverages the Transformer architecture, which is highly effective for sequence-to-sequence tasks.

## Features
* **Transformer Architecture**: Implements multi-head attention and position-wise feed-forward networks.
* **Positional Encoding**: Adds positional information to input embeddings for sequence order.
* **Customizable Hyperparameters**: Adjustable model depth, number of attention heads, and hidden dimensions.
* **Efficient Training**: Implements masking and padding for effective batch processing.
* **Evaluation Metrics**: Supports BLEU score evaluation for translation quality.

## Dataset
The dataset used for training consists of paired sentences in English and Spanish.
Data Preprocessing includes:
* Adding special tokens (`<sos>` and `<eos>`) to target sentences.
* Tokenization and padding of sentences to a maximum length.
* Conversion of tokenized data into PyTorch tensors for training.

## Architecture
The model consists of the following components:

**1. Positional Encoding**
* Uses sine and cosine functions of different frequencies to generate a unique encoding for each position.
* Ensures that the model can recognize the relative positions of tokens in a sequence despite the lack of recurrent or convolutional structures.

**2. Multihead Attention**
* Query (Q), Key (K), and Value (V) matrices are derived from the input data.
* Scaled Dot-Product Attention is used to compute attention scores based on these matrices.
* The attention scores are weighted and used to compute a context vector.
* Multiple Attention Heads: The attention mechanism is split into multiple heads to capture different aspects of the data.
* Linear Transformation: After attention, the results are concatenated and projected back to the original feature space.

**3. Feed Forward Neural Network**
* Consists of two fully connected layers with a ReLU activation and a dropout layer in between.
* Maps the input features from `d_model` dimensions to a higher dimensional space `(d_ff)`, followed by a projection back to `d_model` dimensions.

**4. Encoder**
* **Multihead Self-Attention**: Each token attends to all others in the input sequence.
* **Feed-Forward Network**: Each token's representation is further processed by a feed-forward network.
* **Layer Normalization**: Applied after attention and feed-forward operations to stabilize and accelerate training.
* **Residual Connections**: Adds the input to the output of attention and feed-forward layers to preserve information.
* **Dropout**: Regularization technique to prevent overfitting.

**5. Decoder**
* **Self-Attention**: Each token in the target sequence attends to all previous tokens (masked) to prevent future token leakage during training.
* **Cross-Attention**: Each token in the target sequence attends to all tokens in the encoder’s output to capture context from the input sequence.
* **Feed-Forward Network**: Processes the output of attention mechanisms.
* **Layer Normalization and Residual Connections**: Similar to the encoder, normalization and residual connections are applied to stabilize training and maintain information flow.
* **Dropout**: Reduces overfitting during training.

**6. Transformer**
* **Embedding Layers**: Both input and target sequences are mapped to dense vector representations.
* **Positional Encoding**: Adds position information to the input and target embeddings to allow the model to handle sequence order.
* **Encoder Layers**: A stack of encoder layers, each performing self-attention and feed-forward transformations.
* **Decoder Layers**: A stack of decoder layers, each performing self-attention, cross-attention with encoder output, and feed-forward transformations.
* **Final Linear Layer**: Projects the decoder’s output to the target vocabulary size to generate predictions.
  
## Hyperparameters
Adjust these to customise your training:
* `D_MODEL = 512`
* Sequence Length: `MAX_LENGTH = 25`
* Embedding Dimensions: `EMBEDDING_DIM = 128`
* Number of Attention Heads: `NUM_HEADS = 4`
* Number of Layers of Encoder and Decoder: `NUM_LAYERS = 6`
* Hidden Dimensions: `HIDDEN_DIM = 256`
* Dropout: `DROPOUT = 0.25`
* Batch Size: `BATCH_SIZE = 32`
* Epochs: `NUM_EPOCHS = 50`
* Learning Rate: `LEARNING_RATE = 0.001`
* Gradient Clipping for Stable Training = `CLIP = 1`

## Contributions
Feel free to fork the repository and submit a pull request for improvements.

## License
This project is licensed under the MIT License.
