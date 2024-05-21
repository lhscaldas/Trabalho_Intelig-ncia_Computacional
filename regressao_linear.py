import numpy as np
import matplotlib.pyplot as plt
from perceptron import Target, Dataset, Perceptron2D

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
    lista_target = list()
    lista_linear = list()
    for _ in range(1000):
        # Criar a função target
        target = Target()
        a, b = target.generate_random_line()
        # Criar o dataset
        dataset = Dataset(num_points)
        data, labels = dataset.generate_dataset(target)
        # Criar e treinar o classificador linear
        linear = Linear()
        w = linear.fit(data,labels)
        # Classificar os pontos
        y_predicted = linear.classificar(data)
        # Calcular E_in para essa execução
        lista_E_in.append(np.mean(labels != y_predicted))
        # Guardar para saida
        lista_target.append(target)
        lista_linear.append(linear)
    E_in = np.mean(lista_E_in)
    if verbose: print(f"E_in = {E_in:.4f}")
    return E_in, lista_target, lista_linear # target e linear para usar no E_out

def calc_E_out(num_points, lista_target, lista_linear, verbose = True):
    lista_E_out = list()
    for target, linear in zip(lista_target, lista_linear):
        # Criar o dataset com a mesma função target do E_in
        dataset = Dataset(num_points)
        data, labels = dataset.generate_dataset(target)
        # Classificar os pontos com a mesma hipotese do E_in
        y_predicted = linear.classificar(data)
        # Calcular E_out para essa execução
        lista_E_out.append(np.mean(labels != y_predicted))
    E_out = np.mean(lista_E_out)
    if verbose: print(f"E_out = {E_out:.4f}")
    return E_out

def calc_PLA_iter(num_points):
    lista_iter = list()
    for _ in range(1000):
        # Criar a função target
        target = Target()
        a, b = target.generate_random_line()
        # Criar o dataset
        dataset = Dataset(num_points)
        data, labels = dataset.generate_dataset(target)
        # Criar e treinar o classificador linear
        linear = Linear()
        w = linear.fit(data,labels)
        # Criar e treinar o perceptron
        perceptron = Perceptron2D(weights=w)
        iter, _ = perceptron.fit(data,labels)
        lista_iter.append(iter)
    print(f"{np.mean(lista_iter)} iterações com desvio padrão {np.std(lista_iter):.4f} (min:{np.min(lista_iter)}, máx:{np.max(lista_iter)})")
    
if __name__ == "__main__":
    # teste()
    # _, target, linear = calc_E_in(num_points=100)
    # calc_E_out(num_points=1000, lista_target=target, lista_linear=linear)
    calc_PLA_iter(num_points=10)
