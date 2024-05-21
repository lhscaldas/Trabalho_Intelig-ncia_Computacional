import numpy as np
import matplotlib.pyplot as plt
from perceptron import Target, Dataset

# Classe para criar e treinar o classificador linear
class Linear():
    def __init__(self):
        self.w = np.zeros(3)  # inicializa os pesos (incluindo o w_0)
    
    # Método para calcular a matriz X
    def calc_matriz_X(self, data):
        N = 5
        n_samples = len(data)
        X = np.hstack([np.ones((n_samples, 1)), data]) # adiciona coluna de 1s
        return X
    
    # Método para treinar o classificador linear
    def fit(self, data, labels):
        X = self.calc_matriz_X(data)
        y = labels
        X_T = np.transpose(X)
        X_pseudo_inv = np.dot(np.linalg.inv(np.dot(X_T, X)), X_T)
        self.w = np.dot(X_pseudo_inv, y)
        return self.w
    
    def classificar(self, data):
        X = self.calc_matriz_X(data)
        w_T = np.transpose(self.w)
        y_predicted = np.array([np.sign(np.dot(w_T, xn)) for xn in X])
        return y_predicted
    
def teste():
    # Criar a função target
    target = Target()
    a, b = target.generate_random_line()
    # Criar o dataset e a função target
    num_points = 100
    dataset = Dataset(num_points)
    data, labels = dataset.generate_dataset(target)
    # Criar e treinar o classificador linear
    linear = Linear()
    w = linear.fit(data,labels)
    # Plotar resultados
    plt.figure(figsize=(8, 6))
    x_pos = [data[i][0] for i in range(len(data)) if labels[i] == 1]
    y_pos = [data[i][1] for i in range(len(data)) if labels[i] == 1]
    x_neg = [data[i][0] for i in range(len(data)) if labels[i] == -1]
    y_neg = [data[i][1] for i in range(len(data)) if labels[i] == -1]
    plt.scatter(x_pos, y_pos, c='blue', label='+1')
    plt.scatter(x_neg, y_neg, c='red', label='-1')
    x = np.linspace(-1, 1, 100)
    y_target = a*x+b
    y_g = -(w[1] * x + w[0]) / w[2]
    plt.plot(x, y_g, 'g-', label='Hipótese (g)')
    plt.plot(x, y_target, 'k-', label='Função Target (f)')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Base de dados com o Target (f) e a Hipótese (g)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.grid(True)
    plt.show()

def calc_E_in(num_points, verbose = True):
    lista_E_in = list()
    for _ in range(1000):
        # Criar a função target
        target = Target()
        a, b = target.generate_random_line()
        # Criar o dataset e a função target
        dataset = Dataset(num_points)
        data, labels = dataset.generate_dataset(target)
        # Criar e treinar o classificador linear
        linear = Linear()
        w = linear.fit(data,labels)
        # Classificar os pontos
        y_predicted = linear.classificar(data)
        # Medir E_in
        lista_E_in.append(np.mean(labels != y_predicted))
    E_in = np.mean(lista_E_in)
    if verbose: print(f"E_in = {E_in:.4f}")
    return E_in
    
if __name__ == "__main__":
    # teste()
    calc_E_in(num_points=100)