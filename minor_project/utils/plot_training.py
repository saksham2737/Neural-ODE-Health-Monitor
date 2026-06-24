import matplotlib.pyplot as plt

losses = [0.784, 0.739, 0.697]

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()