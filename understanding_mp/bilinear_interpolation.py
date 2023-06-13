import numpy as np
import cv2

f = np.arange(1, 10).reshape(3,3).astype(np.float32)

res = cv2.resize(f, dsize=(5, 5), interpolation=cv2.INTER_LINEAR)

print(res)