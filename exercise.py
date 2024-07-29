#prior probability
import numpy as np
def create_train_data():
    data=[['Sunny','Hot', 'High', 'Weak', 'no'],
        ['Sunny','Hot', 'High', 'Strong', 'no'],
        ['Overcast','Hot', 'High', 'Weak', 'yes'],
        ['Rain','Mild', 'High', 'Weak', 'yes'],
        ['Rain','Cool', 'Normal', 'Weak', 'yes'],
        ['Rain','Cool', 'Normal', 'Strong', 'no'],
        ['Overcast','Cool', 'Normal', 'Strong', 'yes'],
        ['Overcast','Mild', 'High', 'Weak', 'no'],
        ['Sunny','Cool', 'Normal', 'Weak', 'yes'],
        ['Rain','Mild', 'Normal', 'Weak', 'yes']
        ]
    return np.array(data)
train_data = create_train_data()

def  compute_prior_probability(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probability[i] = len(np.where(train_data[:, -1] == y_unique[i])[0]) / len(train_data)
    return prior_probability

prior_probablity = compute_prior_probability(train_data)
print("P(“Play Tennis” = No)", prior_probablity[0])
print("P(“Play Tennis” = Yes)", prior_probablity[1])

#conditional probability
def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []
    for i in range(train_data.shape[1]-1):
        x_unique = np.unique(train_data[:, i])
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for j in range(len(y_unique)):
            for k in range(len(x_unique)):
                x_conditional_probability[j, k] = len(np.where((train_data[:, i] == x_unique[k]) & (train_data[:, -1] == y_unique[j]))[0])
        conditional_probability.append(x_conditional_probability)
        list_x_name.append(x_unique)
    return conditional_probability, list_x_name

compute_conditional_probability(train_data)

#train naive bayes
def train_naive_bayes(train_data):
    prior_probablity = compute_prior_probability(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(train_data)
    return prior_probablity, conditional_probability, list_x_name

#get index
def get_index(f_name, list_features):
    return np.where(list_features == f_name)[0][0]

#predict
def predict(x, list_x_name, prior_probability, conditional_probability):
    x1 = get_index(x[0], list_x_name[0])
    x2 = get_index(x[1], list_x_name[1])
    x3 = get_index(x[2], list_x_name[2])
    x4 = get_index(x[3], list_x_name[3])
    p0=prior_probability[0] \
    *conditional_probability[0][0,x1] \
    *conditional_probability[1][0,x2] \
    *conditional_probability[2][0,x3] \
    *conditional_probability[3][0,x4]

    p1=prior_probability[1]\
    *conditional_probability[0][1,x1]\
    *conditional_probability[1][1,x2]\
    *conditional_probability[2][1,x3]\
    *conditional_probability[3][1,x4]

    if p0>p1:
        return 'no'
    else:
        return 'yes'
