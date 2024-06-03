from ..autocompressors.auto_compressor import AutoCompressorModel

from transformers import AutoTokenizer


# Load a model pre-trained on 6k tokens in 4 compression steps
tokenizer = AutoTokenizer.from_pretrained("/data1/gj/autocompressor")
model = AutoCompressorModel.from_pretrained("/data1/gj/autocompressor").eval()