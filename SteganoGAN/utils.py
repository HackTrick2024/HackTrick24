import torch
from collections import Counter
from reedsolo import RSCodec
import zlib
from decoders import DenseDecoder
from critics import BasicCritic

import torch
from torch.optim import Adam

import warnings 
warnings.filterwarnings('ignore')

rs = RSCodec(100) # DO NOT CHANGE THIS LINE

METRIC_FIELDS = [
        'val.encoder_mse',
        'val.decoder_loss',
        'val.decoder_acc',
        'val.cover_score',
        'val.generated_score',
        'val.ssim',
        'val.psnr',
        'val.bpp',
        'train.encoder_mse',
        'train.decoder_loss',
        'train.decoder_acc',
        'train.cover_score',
        'train.generated_score',
]

data_depth = 4
hidden_size = 32

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOAD_MODEL = True
PRE_TRAINED_MODEL_PATH = 'image_models/DenseEncoder_DenseDecoder_0.042_2020-07-23_02_08_27.dat'


decoder = DenseDecoder(data_depth, hidden_size).to(device)
critic = BasicCritic(hidden_size).to(device)
cr_optimizer = Adam(critic.parameters(), lr=1e-4)
metrics = {field: list() for field in METRIC_FIELDS}

if LOAD_MODEL: 
    if device == 'cuda':
        checkpoint = torch.load(PRE_TRAINED_MODEL_PATH)
    else:
        checkpoint = torch.load(PRE_TRAINED_MODEL_PATH, map_location=lambda storage, loc: storage)
            
critic.load_state_dict(checkpoint['state_dict_critic'])
decoder.load_state_dict(checkpoint['state_dict_decoder'])
cr_optimizer.load_state_dict(checkpoint['cr_optimizer'])

metrics = checkpoint['metrics']
ep = checkpoint['train_epoch']
date = checkpoint['date']

def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    
    return result


def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))
    
    return bytearray(ints)


def text_to_bytearray(text):
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    return x


def bytearray_to_text(x):
    """Apply error correction and decompress"""
    try:
        text = rs.decode(x)[0]
        zobj = zlib.decompressobj()  # obj for decompressing data streams that won’t fit into memory at once.
        text = zobj.decompress(text, zlib.MAX_WBITS|32)

        return text.decode("utf-8")

    except BaseException as e:
        return False


def make_payload(width, height, depth, text):
    """
    This takes a piece of text and encodes it into a bit vector. It then
    fills a matrix of size (width, height) with copies of the bit vector.
    """
    message = text_to_bits(text) + [0] * 32

    payload = message
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]

    return torch.FloatTensor(payload).view(1, depth, height, width)

def make_message(image):
    image = image.to(device)

    image = decoder(image).view(-1) > 0
    image = torch.tensor(image, dtype=torch.uint8)

    # split and decode messages
    candidates = Counter()
    bits = image.data.cpu().numpy().tolist()
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
        candidate = bytearray_to_text(bytearray(candidate))
        if candidate:
            candidates[candidate] += 1

    # choose most common message
    if len(candidates) == 0:
    #   raise ValueError('Failed to find message.')
        return

    candidate, _ = candidates.most_common(1)[0]
    return candidate


def decode(generated: torch.Tensor) -> str:
    text_return = make_message(generated)
    return text_return