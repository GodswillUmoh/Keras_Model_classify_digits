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

### Print X_train
```python
print(X_train.shape)
```

### Building the model
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation= 'relu'),
    keras.layers.Dense(10, activation= 'softmax')
])
```

### Compile
```python
model.compile( optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
```

### Train model with 10 epochs
```python
model.fit(X_train, y_train, epochs= 10)
```

### Model Summary
```python
model.summary()
```

### Evaluate model
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
```

### Print Accuracy
```python
print(f'Test Accuracy {test_acc}')
```

[To see code in the terminal, kindly click here](https://colab.research.google.com/drive/19U84C4of-dOWt1_tCYImCY8HC2T-jHpA#scrollTo=BvTIEijhtJ7s)
