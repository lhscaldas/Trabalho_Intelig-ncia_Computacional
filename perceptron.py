import numpy as np
import matplotlib.pyplot as plt

# Classe para criar a função target
class Target:
    def __init__(self): 
        self.a = 0 # coeficiente angular
        self.b = 0 # coeficiente linear

    # Método para gerar a linha da função target
    def generate_random_line(self):
        point1 = np.random.uniform(-1, 1, 2) # ponto aleatorio no domínio
        point2 = np.random.uniform(-1, 1, 2) # ponto aleatorio no domínio
        a = (point2[1] - point1[1]) / (point1[0] - point2[0]) # cálculo do coeficiente angular
        b = point1[1] - a*point1[0] # cálculo do coeficiente linear
        self.a = a
        self.b = b
        return a, b
    
    # Método para classificar pontos de acordo a função target
    def classify_point(self, point):
        a = self.a
        b = self.b
        y_reta = a*point[0] + b    
        return np.sign(point[1] - y_reta) # verifica se a coordenada y do ponto está acima ou abaixo da reta

# Classe para criar o dataset
class Dataset:
    def __init__(self, N): 
        self.N = N # tamanho do dataset

    # Método para gerar a base de dados D
    def generate_dataset(self, target):
        N = self.N
        data = np.random.uniform(-1, 1, (N, 2)) # gera N pontos no R2 com coordenadas entre [-1, 1]
        labels = np.array([target.classify_point(point) for point in data])
        return data, labels

# Classe para criar e treinar o perceptron 2D
class Perceptron2D:
    def __init__(self, weights = np.zeros(3)):
        self.w = weights # inicializa os pesos (incluindo o w_0)
    
    # Método para treinar o perceptron usando o algoritmo de aprendizagem perceptron (PLA)
    def pla(self, data, labels): 
        n_samples = len(data)
        X_bias = np.hstack([np.ones((n_samples, 1)), data]) # adiciona uma coluna de 1s para o X_0 (coordenada artificial)
        iterations = 0
        while True:
            predictions = np.sign(np.dot(X_bias, self.w))
            errors = labels != predictions
            if not np.any(errors):
                break
            misclassified_indices = np.where(errors)[0]
            random_choice = np.random.choice(misclassified_indices)
            self.w += labels[random_choice] * X_bias[random_choice] # atualiza os pesos
            iterations += 1
        return iterations, self.w
    
    # Método para classificar um dataset com base nos pesos aprendidos.
    def classificar(self, data):
        n_samples = len(data)
        X_bias = np.hstack([np.ones((n_samples, 1)), data]) # adiciona uma coluna de 1s para o bias X_0
        return np.sign(np.dot(X_bias, self.w)) # verifica o sinal do produto escalar entre x e w

def scatterplot(data, labels, target, hipotese):
    a, b = target.a, target.b
    w = hipotese.w
    plt.figure(figsize=(8, 6))
    # plotar a função target
    x = np.linspace(-1, 1, 100)
    y_target = a*x+b
    plt.plot(x, y_target, 'k-', label='Função Target (f)')
    # plotar a hipótese
    y_g = -(w[1] * x + w[0]) / w[2]
    plt.plot(x, y_g, 'g-', label='Hipótese (g)')
    # plotar os pontos
    x_pos = [data[i][0] for i in range(len(data)) if labels[i] == 1]
    y_pos = [data[i][1] for i in range(len(data)) if labels[i] == 1]
    x_neg = [data[i][0] for i in range(len(data)) if labels[i] == -1]
    y_neg = [data[i][1] for i in range(len(data)) if labels[i] == -1]
    plt.scatter(x_pos, y_pos, c='blue', label='+1')
    plt.scatter(x_neg, y_neg, c='red', label='-1')
    # ajustar a figura       
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Base de dados com a Função Target (f) e a Hipótese (g)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.grid(True)
    plt.show()

def teste():
    # Criar a função target
    target = Target()
    a, b = target.generate_random_line()
    # Criar o dataset
    num_points = 100
    dataset = Dataset(num_points)
    data, labels = dataset.generate_dataset(target)
    # Criar e treinar o perceptron
    perceptron = Perceptron2D()
    _, w = perceptron.pla(data,labels)
    # Plotar resultados
    scatterplot(data, labels, target, perceptron)

def calc_num_iter(num_points, verbose = True):
    lista_iter = list()
    for _ in range(1000):
        target = Target()
        target.generate_random_line()
        dataset = Dataset(num_points)
        data, labels = dataset.generate_dataset(target)
        perceptron = Perceptron2D()
        iter, _ = perceptron.pla(data,labels)
        lista_iter.append(iter)
    if verbose: print(f"{np.mean(lista_iter)} iterações com desvio padrão {np.std(lista_iter):.4f} (min:{np.min(lista_iter)}, máx:{np.max(lista_iter)})")
    return lista_iter

def calc_p_erro(num_points, verbose = True):
    lista_erro = list()
    for _ in range(1000):
        target = Target()
        target.generate_random_line()
        dataset_train = Dataset(num_points)
        x_train, y_train = dataset_train.generate_dataset(target)
        dataset_test = Dataset(10000) # mais 10mil pontos
        x_test, y_test = dataset_test.generate_dataset(target)
        perceptron = Perceptron2D()
        perceptron.pla(x_train,y_train)
        y_predicted = perceptron.classificar(x_test)
        erro = np.mean(y_test != y_predicted)
        lista_erro.append(erro)
    if verbose: print(f"P[f(x)\u2260g(x)] = {np.mean(lista_erro):.4f}")
    return lista_erro

def relationship(lista_num_points):
    lista_iter_medio = list()
    lista_erro_medio = list()
    for num_points in lista_num_points:
        lista_iter = calc_num_iter(num_points, verbose=False)
        lista_erro = calc_p_erro(num_points, verbose=False)
        lista_iter_medio.append(np.mean(lista_iter))  
        lista_erro_medio.append(np.mean(lista_erro))
    return lista_iter_medio, lista_erro_medio

def plot_relationship(num_points_list, lista_iter_medio, lista_erro_medio):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(num_points_list,lista_iter_medio,c="blue",label="Iterações")
    ax.set_title("Curvas de relação")
    ax.set_xlabel('Nº de pontos')
    ax.set_ylabel('Iterações')
    ax2=ax.twinx()
    ax2.plot(num_points_list,lista_erro_medio,c="red", label='P[f(x)\u2260g(x)]')
    ax2.set_ylabel('P[f(x)\u2260g(x)]')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.9))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # teste()
    calc_num_iter(num_points = 100)
    # calc_p_erro(num_points = 10)
    # num_points_list = np.arange(10, 101, 50, dtype=int)
    # lista_iter_medio, lista_erro_medio = relationship(num_points_list)
    # plot_relationship(num_points_list, lista_iter_medio, lista_erro_medio)


    

   