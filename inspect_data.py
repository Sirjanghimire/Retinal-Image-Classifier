from pathlib import Path
from medmnist import OCTMNIST
import matplotlib.pyplot as plt

Path("data").mkdir(exist_ok=True)

dataset = OCTMNIST(
    root="data",
    split="train",
    download=True,
    size=64
)

image, label = dataset[0]
print(len(dataset))
print(image.size)
print(label)

plt.imshow(image, cmap="grey")
plt.title("First Training Image")
plt.show()
