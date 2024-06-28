import random
import constraint
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
import tensorflow as tf

class CSPLoss(Loss):
    def __init__(self, variables, constraints, **kwargs):
        super().__init__(**kwargs)
        self.variables = variables
        self.constraints = constraints
        print(constraints)

    def call(self, y_true, y_pred):
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        y_true = tf.reshape(y_true, [-1, num_variables, domain_size])
        y_pred = tf.reshape(y_pred, [-1, num_variables, domain_size])

        y_pred_decoded = tf.argmax(y_pred, axis=-1)
        y_true_decoded = tf.argmax(y_true, axis=-1)

        penalty = 0.0

        for constraints_list in self.constraints:
            for constraint in constraints_list:
                var1, var2, relation = constraint
                if var1 in self.variables and var2 in self.variables:
                    var1_index = self.variables.index(var1)
                    var2_index = self.variables.index(var2)

                    if relation == '!=':
                        penalty += tf.reduce_sum(
                            tf.cast(y_pred_decoded[:, var1_index] == y_pred_decoded[:, var2_index], tf.float32))

                    elif relation == '==':
                        penalty += tf.reduce_sum(
                            tf.cast(y_pred_decoded[:, var1_index] == y_pred_decoded[:, var2_index], tf.float32))

        penalty_weight = 0.01
        total_loss = base_loss + penalty_weight * penalty

        return total_loss


def generate_csp_instance(num_variables, domain_size, num_constraints, predicted_problem=False, unsolvable_chance=0.5):
    problem = constraint.Problem()
    variables = [f'X{i}' for i in range(num_variables)]
    domains = list(range(domain_size))
    constraints = []

    for var in variables:
        problem.addVariable(var, domains)

    for _ in range(num_constraints):
        var1, var2 = random.sample(variables, 2)
        problem.addConstraint(lambda x, y: x != y, (var1, var2))
        constraints.append((var1, var2, '!='))
        if predicted_problem:
            print(var1 + " != " + var2)

    if random.random() < unsolvable_chance:
        # Add an unsolvable constraint
        var1, var2 = random.sample(variables, 2)
        problem.addConstraint(lambda x, y: x == y, (var1, var2))
        constraints.append((var1, var2, '=='))
        if predicted_problem:
            print(var1 + " = " + var2)

    return problem, variables, constraints

def solve_and_label_csp(problem, variables):
    solutions = problem.getSolutions()
    if solutions:
        return {'solvable': True, 'solution': solutions[0]}
    else:
        return {'solvable': False, 'solution': None}

def generate_mixed_dataset(num_instances, num_variables, domain_size, num_constraints, unsolvable_chance=0.5):
    dataset = []
    for _ in range(num_instances):
        problem, variables, constraints = generate_csp_instance(num_variables, domain_size, num_constraints, False, unsolvable_chance)
        label = solve_and_label_csp(problem, variables)
        dataset.append((problem, variables, label, constraints))
    return dataset

def csp_to_input(problem, variables):
    num_variables = len(variables)
    domain_size = len(problem._variables[variables[0]])
    input_data = np.zeros((num_variables, domain_size))
    return input_data

def solution_to_output(label, num_variables, domain_size):
    if label['solvable']:
        output_data = np.array([label['solution'][var] for var in sorted(label['solution'])])
        output_data = to_categorical(output_data, num_classes=domain_size)
    else:
        output_data = np.full((num_variables, domain_size), -1)  # Use -1 or some other indicator for unsolvable
    return output_data, label['solvable']

num_instances = 5000
num_variables = 5
domain_size = 5
num_constraints = 5

dataset = generate_mixed_dataset(num_instances, num_variables, domain_size, num_constraints)

inputs = []
outputs = []
solvable_labels = []
constraints_list = []
for problem, variables, label, constraints in dataset:
    input_data = csp_to_input(problem, variables)
    output_data, solvable = solution_to_output(label, len(variables), domain_size)
    inputs.append(input_data)
    outputs.append(output_data)
    solvable_labels.append(solvable)
    constraints_list.append(constraints)

inputs = np.array(inputs)
outputs = np.array(outputs)
solvable_labels = np.array(solvable_labels)

input_layer = Input(shape=(num_variables * domain_size,))
dense1 = Dense(256, activation='relu')(input_layer)
dense2 = Dense(256, activation='relu')(dense1)

output_solution = Dense(num_variables * domain_size, activation='softmax')(dense2)

output_solvable = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=input_layer, outputs=[output_solution, output_solvable])
model.compile(optimizer=Adam(),
              loss=[CSPLoss(variables, constraints_list), 'binary_crossentropy'],
              metrics=['accuracy', 'accuracy'])

inputs_flat = inputs.reshape((inputs.shape[0], num_variables * domain_size))

model.fit(inputs_flat, [outputs.reshape((-1, num_variables * domain_size)), solvable_labels], epochs=500, batch_size=32)


# input_layer = Input(shape=(num_variables, domain_size, 1))
#

# conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
# flatten = Flatten()(conv1)
#

# dense_solution = Dense(128, activation='relu')(flatten)
# output_solution = Dense(num_variables * domain_size, activation='softmax')(dense_solution)
# output_solution = Reshape((num_variables, domain_size))(output_solution)
#
# dense_solvable = Dense(128, activation='relu')(flatten)
# output_solvable = Dense(1, activation='sigmoid')(dense_solvable)
#
# model = Model(inputs=input_layer, outputs=[output_solution, output_solvable])
# model.compile(optimizer='adam',
#               loss=['categorical_crossentropy', 'binary_crossentropy'],
#               metrics=[['accuracy'], ['accuracy']])
#
# inputs = inputs.reshape((inputs.shape[0], num_variables, domain_size, 1))
#
# model.fit(inputs, [outputs, solvable_labels], epochs=50, batch_size=32)


new_problem, new_variables, new_constraints = generate_csp_instance(num_variables, domain_size, num_constraints, True)
input_data = csp_to_input(new_problem, new_variables)
input_data = input_data.reshape((1, num_variables * domain_size))

print("Generated CSP Problem and Variables:")
print(new_problem)
print(new_variables)

# Predict and print the solution
predicted_solution, predicted_solvability = model.predict(input_data)

predicted_solvability = bool(round(predicted_solvability[0][0]))
print(f"Predicted Solvability: {predicted_solvability}")

if predicted_solvability:
    predicted_solution = predicted_solution[0].reshape((num_variables, domain_size))
    solution_dict = {f'X{i}': int(np.argmax(predicted_solution[i])) for i in range(len(predicted_solution))}
    print(f"Predicted Solution: {solution_dict}")
else:
    print("The CSP instance is predicted to be unsolvable.")
