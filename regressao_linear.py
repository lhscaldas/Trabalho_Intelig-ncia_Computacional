import numpy as np
import matplotlib.pyplot as plt
from perceptron import Target, Dataset, Perceptron2D, scatterplot

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
    
# Classe Perceptron2D com algoritmo pocket
class Perceptron2Dmod(Perceptron2D):

    # Método para treinar o perceptron usando o algoritmo Pocket
    def pocket(self, data, labels, max_iter = 1000): 
        n_samples = len(data)
        X_bias = np.hstack([np.ones((n_samples, 1)), data]) # adiciona uma coluna de 1s para o X_0 (coordenada artificial)
        iterations = 0
        E_in = 1
        w = self.w
        while E_in > 0 and iterations < max_iter:
            E_in_atual = 0
            w_atual = w
            errors = 0
            for i in range(n_samples):
                if labels[i] * np.dot(self.w, X_bias[i]) <= 0:
                    w_atual += labels[i] * X_bias[i] # atualiza os pesos
                    errors += 1
            iterations += 1
            E_in_atual = errors/n_samples
            if E_in_atual < E_in: 
                E_in = E_in_atual
                w = w_atual
        self.w = w
        return iterations, self.w, E_in
    
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
    scatterplot(data, labels, target, linear)

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
        target.generate_random_line()
        # Criar o dataset
        dataset = Dataset(num_points)
        data, labels = dataset.generate_dataset(target)
        # Criar e treinar o classificador linear
        linear = Linear()
        w = linear.fit(data,labels)
        # Criar e treinar o perceptron
        perceptron = Perceptron2D(weights=w)
        iter, _ = perceptron.pla(data,labels)
        lista_iter.append(iter)
    print(f"{np.mean(lista_iter)} iterações com desvio padrão {np.std(lista_iter):.4f} (min:{np.min(lista_iter)}, máx:{np.max(lista_iter)})")

def calc_pocket_E_out(N1, N2):
    lista_E_in = list()
    lista_E_out = list()
    for _ in range(1000):
        # Criar a função target
        target = Target()
        a, b = target.generate_random_line()
        # Criar o dataset de treinamento
        dataset_train = Dataset(N1)
        x_train, y_train = dataset_train.generate_dataset(target)
        selected_indices = np.random.choice(len(y_train), int(len(y_train) * 0.1), replace=False) # seleciona 10%
        y_train[selected_indices] *= -1 # inverte o valor de 10%
        # Criar e treinar o classificador linear
        linear = Linear()
        w = linear.fit(x_train,y_train)
        # Criar e treinar o perceptron com pocket
        perceptron_mod = Perceptron2Dmod(weights=w)
        _, _, E_in = perceptron_mod.pocket(x_train,y_train)
        lista_E_in.append(E_in)
        # Criar o dataset de teste
        dataset_test = Dataset(N2)
        x_test, y_test = dataset_test.generate_dataset(target)
        # Classificar os pontos com a mesma hipotese do E_in
        y_predicted = perceptron_mod.classificar(x_test)
        # Calcular E_out para essa execução
        E_out = np.mean(y_test != y_predicted)
        lista_E_out.append(E_out)
    # Printar E_in e E_out médios
    print(f"E_in = {np.mean(E_in):.4f}")
    print(f"E_out = {np.mean(E_out):.4f}")
    # Plotar o dataset, a função target e a hipótese g da última execução
    scatterplot(x_train, y_train, target, perceptron_mod)

    
if __name__ == "__main__":
    # teste()
    # _, target, linear = calc_E_in(num_points=100)
    # calc_E_out(num_points=1000, lista_target=target, lista_linear=linear)
    # calc_PLA_iter(num_points=10)
    calc_pocket_E_out(N1 = 100, N2 = 1000)
