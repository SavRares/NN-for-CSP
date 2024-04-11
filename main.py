import numpy as np
import tensorflow as tf

def generate_binary_csp(n, p, q):
    # CSP binar folosind Model A
    csp_matrix = np.random.choice([0, 1], size=(n, n), p=[1 - p, p])
    # qij pentru fiecare pereche
    q_values = np.random.uniform(0, q, size=(n, n))
    np.fill_diagonal(q_values, 0)
    for i in range(n):
        for j in range(i + 1, n):
            if csp_matrix[i][j] == 1:
                csp_matrix[i][j] = np.random.choice([0, 1], p=[q_values[i][j], 1 - q_values[i][j]])
                csp_matrix[j][i] = csp_matrix[i][j]  # Compatibilitatea intre doua variabile e simetrica

    return csp_matrix


def label_csp(csp_matrix, p, q):
    # Label pentru CSP cu SATISFIABLE sau UNSATISFIABLE
    if np.random.rand() < p:
        solution = np.random.randint(2, size=csp_matrix.shape[0])
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                if solution[i] == solution[j]:
                    csp_matrix[i][j] = 1
                    csp_matrix[j][i] = 1
    else:
        pass

    return csp_matrix


def generate_labeled_data(n_samples, n_variables, p, q):
    # Generare data cu label-uri folosind GMAM
    data_points = []
    labels = []
    for _ in range(n_samples):
        csp_matrix = generate_binary_csp(n_variables, p, q)
        labeled_csp_matrix = label_csp(csp_matrix, p, q)
        data_points.append(labeled_csp_matrix)
        labels.append(1 if np.random.rand() < p else 0)  # 1 pentru SATISFIABLE, 0 pentru UNSATISFIABLE

    return np.array(data_points), np.array(labels)

# Antrenament
n_samples = 10000
n_variables = 10
p = 0.5
q = 0.5
data, labels = generate_labeled_data(n_samples, n_variables, p, q)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(n_variables, n_variables, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Assuming binary classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(data, labels, epochs=10, validation_split=0.2)

# Predictii pentru data noua
n_samples_prediction = 1
n_variables_prediction = 10
p_prediction = 0.5
q_prediction = 0.5

new_data_points, _ = generate_labeled_data(n_samples_prediction, n_variables_prediction, p_prediction, q_prediction)
print(new_data_points)
predictions = model.predict(new_data_points)
print(predictions)
