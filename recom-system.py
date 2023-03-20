import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# wemporting the dataset
print('wemporting Dataset =====>')
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


# Preparing the training set and the test set
print('Reading testing and training datasets =====>')


# Training and test set for 100k users
training_set_df = pd.read_csv('ml-100k/u1.base', delimiter = '\t', header=None)
test_set_df = pd.read_csv('ml-100k/u1.test', delimiter = '\t', header=None)

# Convert training set and test set in numpy arrays
training_set_ar = np.array(training_set_df, dtype = 'int')
test_set_ar = np.array(test_set_df, dtype = 'int')
# Getting the number of users and movies
nb_users = int(max(max(training_set_ar[:,0]), max(test_set_ar[:,0])))
nb_movies = int(max(max(training_set_ar[:,1]), max(test_set_ar[:,1])))
nb_userAttributes = 4
users['female_user'] = (users[1] == 'F').astype(int)
users['male_user'] = (users[1] == 'M').astype(int)

# extract unique genre values
print('Extracting unique genres =====>')
genre = movies[2]
unique_genre = genre.unique()
genre_values = []
for movie_genre in unique_genre:
    mg = movie_genre.split("|")
    for g in mg:
        if g not in genre_values:
            genre_values.append(g)
           
genre_values = sorted(genre_values, key=str.lower)
print(genre_values)
print(len(genre_values))

def get_genre_vector(genre_row_val):
    mg = genre_row_val.split("|")
    gen_vec = np.zeros(len(genre_values))
    gen_index = 0
    for g in genre_values:
        if g in mg:
            gen_vec[gen_index] = 1
        gen_index += 1
return gen_vec
# unit tests for above function
'''print(get_genre_vector("Action|Adventure|Romance"))
print(get_genre_vector("Animation|Children's|Comedy"))
print(get_genre_vector("Thriller"))
print(get_genre_vector("Animation|Children's|Comedy|Romance"))'''

# Add Genre Vector to movies dataframe
print('Creating Genre vector on movies df ====>')
movie_data = movies[2]
movie_col = []
gen_index = 0
for movie_gen in movie_data:
    gen_vec = get_genre_vector(movie_gen)
    movie_col.append(gen_vec)
    gen_index += 1
   
movies['genre_vector'] = movie_col

def addgenrevector(data):
    genre_array = []
    movie_id_list = data[1].tolist()
    for movie_id in movie_id_list:
        genre_array.append(movies.loc[movies[0] == movie_id]['genre_vector'])
    data['genre_vector'] = genre_array
    return data
       
print('Adding Genre Vector to training and testing datasets =====>')
training_set_gen_df = addgenrevector(training_set_df)
training_set_gen_ar = np.array(training_set_gen_df)
test_set_gen_df = addgenrevector(test_set_df)
test_set_gen_ar = np.array(test_set_gen_df)  
def createmultidimensionalmatrix(data):
    print(data.shape)
    gen_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[1][data[0] == id_users]
        id_ratings = data[2][data[0] == id_users]
        user_genre_list = data['genre_vector'][data[0] == id_users][data[2] >= 3]
        female_user = float(users['female_user'][users[0] == id_users])
        male_user = float(users['male_user'][users[0] == id_users])
        user_age = float(users[2][users[0] == id_users])
        reg_months = float(users[3][users[0] == id_users])
        user_genre_sum = np.zeros(len(genre_values))
        for usr_gen_vec in user_genre_list:
            if len(usr_gen_vec):
                user_genre_sum = user_genre_sum + np.array(usr_gen_vec)
        data_reshaped = np.zeros(nb_movies)
        # Create a matrix with users in rows and ratings for each movie in columns
        data_reshaped[id_movies - 1] = id_ratings
        # Add columns of user genre only for good ratings
        if user_genre_sum[0].shape:
            data_reshaped = np.append(data_reshaped, user_genre_sum[0])
        else:
            data_reshaped = np.append(data_reshaped, user_genre_sum)
           
        data_reshaped = np.append(data_reshaped, [female_user])
        data_reshaped = np.append(data_reshaped, [male_user])
        data_reshaped = np.append(data_reshaped, [user_age])
        data_reshaped = np.append(data_reshaped, [reg_months])
        gen_data.append(list(data_reshaped))
    return gen_data
       
       
print('Creating 2D matrix ======>')    
training_gen_data = createmultidimensionalmatrix(training_set_gen_df)
test_gen_data = createmultidimensionalmatrix(test_set_gen_df)



# Converting the data into Torch tensors
print('Creating torch tensors ======>')
training_set_1 = torch.FloatTensor(training_gen_data)
test_set_1 = torch.FloatTensor(test_gen_data)

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(input_columns, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, input_columns)
        self.activation = nn.Sigmoid()
       
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)



nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set_1[id_user]).unsqueeze(0)
        target = input.clone()
        #Select only rating related columns to compute loss
        target_ratings = target[:, :nb_movies]
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            output_ratings = output[:, :nb_movies]
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output_ratings, target_ratings)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
   
   
# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set_1[id_user]).unsqueeze(0)
    target = Variable(test_set_1[id_user]).unsqueeze(0)
    target_ratings = target[:, :nb_movies]
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        output_ratings = output[:, :nb_movies]
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output_ratings, target_ratings)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))