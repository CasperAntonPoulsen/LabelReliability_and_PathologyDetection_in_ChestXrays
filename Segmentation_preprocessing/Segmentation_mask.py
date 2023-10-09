import pandas as pd
import numpy as np

files = "/home/data_shares/purrlab/physionet.org/files/chexmask-cxr-segmentation-data/0.2"

padchest_masks = pd.read_csv(files+ "/OriginalResolution/Padchest.csv")
padchest_masks.head()