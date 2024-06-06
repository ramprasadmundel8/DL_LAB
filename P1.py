import numpy as np
import matplotlib.pyplot as plt

def plot_sigmoid():
    x = np.linspace(-10, 10, 100)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y)
    plt.xlabel('Input')
    plt.ylabel('Sigmoid Output')
    plt.title('Sigmoid Activation Function')
    plt.grid(True)
    plt.show()

def plot_tanh():
    x = np.linspace(-10, 10, 100)
    tanh = np.tanh(x)
    plt.plot(x, tanh)
    plt.title("Hyperbolic Tangent (tanh) Activation Function")
    plt.xlabel("x")
    plt.ylabel("tanh(x)")
    plt.grid(True)
    plt.show()

def plot_relu():
    x = np.linspace(-10, 10, 100)
    relu = np.maximum(0, x)
    plt.plot(x, relu)
    plt.title("ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("ReLU(x)")
    plt.grid(True)
    plt.show()

def plot_leaky_relu():
    x = np.linspace(-10, 10, 100)
    def leaky_relu(x, alpha=0.1):
        return np.where(x >= 0, x, alpha * x)
    leaky_relu_values = leaky_relu(x)
    plt.plot(x, leaky_relu_values)
    plt.title("Leaky ReLU Activation Function")
    plt.xlabel("x")
    plt.ylabel("Leaky ReLU(x)")
    plt.grid(True)
    plt.show()

def softmax():
    def softmax_act(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)
    x = np.array([1, 2, 3])
    result = softmax_act(x)
    print(result)
    def plot_softmax(probabilities, class_labels):
        plt.bar(class_labels, probabilities)
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Softmax Output")
        plt.show()
    class_labels = ["Class A", "Class B", "Class C"]
    plot_softmax(result, class_labels)

while True:
    print("\nMAIN MENU")
    print("1. Sigmoid")
    print("2. Hyperbolic tangent")
    print("3. Rectified Linear Unit")
    print("4. Leaky ReLU")
    print("5. Softmax")
    print("6. Exit")
    choice = int(input("Enter the Choice:"))
    if choice == 1:
        plot_sigmoid()
    elif choice == 2:
        plot_tanh()
    elif choice == 3:
        plot_relu()
    elif choice == 4:
        plot_leaky_relu()
    elif choice == 5:
        softmax()
    elif choice == 6:
        break
    else:
        print("Oops! Incorrect Choice.")
