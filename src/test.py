import numpy as np
import pysomemodule

image  = np.ones((100,100), dtype=np.float32)
rectangles = np.random.uniform(20,30, (100,4)).astype(np.float32)

abc = np.zeros((100, 100*3)).astype(np.float32)
print pysomemodule.ABC("").doSomething(image, rectangles, rectangles, abc)
print abc
