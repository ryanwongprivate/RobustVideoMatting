import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from inference_utils import VideoReader, VideoWriter

reader = VideoReader('input.mp4', transform=ToTensor())
writer = VideoWriter('output.mp4', frame_rate=30)

# Load model.
sess = ort.InferenceSession('rvm_mobilenetv3_fp16.onnx')

# Create an io binding.
io = sess.io_binding()

# Create tensors on CUDA.
rec = [ ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=np.float16), 'cuda') ] * 4
downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([0.25], dtype=np.float32), 'cuda')

# Set output binding.
for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
    io.bind_output(name, 'cuda')

# Inference loop
for src in DataLoader(reader):
    io.bind_cpu_input('src', src)
    io.bind_ortvalue_input('r1i', rec[0])
    io.bind_ortvalue_input('r2i', rec[1])
    io.bind_ortvalue_input('r3i', rec[2])
    io.bind_ortvalue_input('r4i', rec[3])
    io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

    sess.run_with_iobinding(io)

    fgr, pha, *rec = io.get_outputs()

    # Only transfer `fgr` and `pha` to CPU.
    fgr = fgr.numpy()
    pha = pha.numpy()
    writer.write(fgr * pha)
