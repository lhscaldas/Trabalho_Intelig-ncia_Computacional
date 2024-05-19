# Método para calcular a divergência entre f e g
def calculate_divergence(a, b, c, weights, num_points=10000):
    test_points = np.random.uniform(-1, 1, (num_points, 2))
    test_points_with_bias = np.hstack((np.ones((num_points, 1)), test_points))
    true_labels = np.array([classify_point(a, b, c, point) for point in test_points])
    predicted_labels = np.sign(np.dot(test_points_with_bias, weights[-1]))
    divergence = np.mean(true_labels != predicted_labels)
    return divergence

# Executar o experimento 1000 vezes
num_experiments = 1000
num_points = 100
iterations_list = []
divergence_list = []

for _ in range(num_experiments):
    a, b, c = generate_random_line()
    data, labels = generate_dataset(num_points, a, b, c)
    weights, iterations = pla(data, labels)
    divergence = calculate_divergence(a, b, c, weights[-1])
    iterations_list.append(iterations)
    divergence_list.append(divergence)
    

# Calcular as médias
average_iterations = np.mean(iterations_list)
average_divergence = np.mean(divergence_list)

print(f"Média de iterações para convergência: {average_iterations}")
print(f"Média de divergência entre f e g: {average_divergence}")