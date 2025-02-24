# Keras_Model_classify_digits

## Practical Example (Handwritten Digit Recognition with TensorFlow)

### Task
> We will build a model that recognizes handwritten digits (0-9) using MNIST dataset.

## Python: Keras Code

### Importing the dependencies
```python
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
```

### Load Data
```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

### Normalize the independent Variable
```python
(X_train, X_test) = X_train / 255.0, X_test / 255.0
```


