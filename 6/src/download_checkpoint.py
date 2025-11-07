from bionemo.core.data.load import load


checkpoint_path = load("esm2/650m:2.0")
print("checkpoint_path: ", checkpoint_path)
