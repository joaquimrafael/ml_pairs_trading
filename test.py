import numpy, pandas, matplotlib, darts, torch, tensorflow
print("NumPy    ", numpy.__version__)
print("pandas   ", pandas.__version__)
print("matplotlib", matplotlib.__version__)
print("darts    ", darts.__version__)
print("torch    ", torch.__version__, "CUDA?", torch.cuda.is_available())
print("TF       ", tensorflow.__version__)