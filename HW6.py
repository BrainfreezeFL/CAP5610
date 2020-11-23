import pandas as pd
import math
import numpy as np
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNWithMeans
def find_mean(array) : 
  length = len(array)
  sum = 0
  for items in array : 
    sum = sum + float(items)
  return sum/length

data = pd.read_csv('ratings_small.csv', 
                 dtype= {'userId':np.int32, 
                         'movieId':np.int32, 
                         'rating':np.float64, 
                         'timestamp':np.int64},
                 header=0, #skiprows=1
                 names= ['userId','movieId','rating','timestamp'])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
#print(train_df)


# Load the movielens-100k dataset (download it if needed).
#data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo_probabilistic_matrix_factorization = SVD(biased = False)
algo_user_collab_filter_cosine = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': True})
algo_user_collab_filter_pearson = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo_user_collab_filter_msd = KNNWithMeans(k=50, sim_options={'name': 'MSD', 'user_based': True})
algo_item_collab_filter_pearson = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo_item_collab_filter_cosine = KNNWithMeans(k=50, sim_options={'name': 'cosine', 'user_based': False})
algo_item_collab_filter_msd = KNNWithMeans(k=50, sim_options={'name': 'msd', 'user_based': False})

# Differences in K-value
algo_item_five = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo_item_ten = KNNWithMeans(k=10, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo_item_twenty = KNNWithMeans(k=20, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo_item_fifty = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': False})

algo_user_five = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo_user_ten = KNNWithMeans(k=10, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo_user_twenty = KNNWithMeans(k=20, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo_user_fifty = KNNWithMeans(k=50, sim_options={'name': 'pearson_baseline', 'user_based': True})
user_one = []
for i in range(20):
  print("This run had this many K's")
  print(i)
  algo_user = KNNWithMeans(k=i, sim_options={'name': 'pearson_baseline', 'user_based': True})
  user_one = cross_validate(algo_user, data, measures=['RMSE'], cv=5, verbose=True)
  user_mean = find_mean(user_one["test_rmse"])
  print("The User mean is : ")
  print(user_mean)
  #user_one.append(user_mean)

item = []
for i in range(20):
  print("This run had this many K's")
  print(i)
  algo_item = KNNWithMeans(k=i, sim_options={'name': 'pearson_baseline', 'user_based': False})
  item = cross_validate(algo_item, data, measures=['RMSE'], cv=5, verbose=True)
  item_mean = find_mean(item["test_rmse"])
  print("The mean is : ")
  print(item_mean)
  #item.append(item_mean)

# Run 5-fold cross-validation and print results.
print("PMF")
cross_validate(algo_probabilistic_matrix_factorization, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("UCF-Cosine")
user_cosine = cross_validate(algo_user_collab_filter_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("UCF-Pearson")
user_pearson = cross_validate(algo_user_collab_filter_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("UCF-msd")
user_msd = cross_validate(algo_user_collab_filter_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("ICF-Cosine")
item_cosine = cross_validate(algo_item_collab_filter_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("ICF-Pearson")
item_pearson = cross_validate(algo_item_collab_filter_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("ICF-MSD")
item_msd = cross_validate(algo_item_collab_filter_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

user_five = cross_validate(algo_user_five, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
user_ten = cross_validate(algo_user_ten, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
user_twenty = cross_validate(algo_user_twenty, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
user_fifty = cross_validate(algo_user_fifty, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

item_five = cross_validate(algo_item_five, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
item_ten = cross_validate(algo_item_ten, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
item_twenty = cross_validate(algo_item_twenty, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
item_fifty = cross_validate(algo_item_fifty, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

user_five_rmse = user_five["test_rmse"]
user_ten_rmse = user_ten["test_rmse"]
user_twenty_rmse = user_twenty["test_rmse"]
user_fifty_rmse = user_fifty["test_rmse"]
user_five_mae = user_five["test_mae"]
user_ten_mae = user_ten["test_mae"]
user_twenty_mae = user_twenty["test_mae"]
user_fifty_mae = user_fifty["test_mae"]
user_five_rmse_mean = find_mean(user_five_rmse)
user_ten_rmse_mean = find_mean(user_ten_rmse)
user_twenty_rmse_mean = find_mean(user_twenty_rmse)
user_fifty_mae_mean = find_mean(user_fifty_rmse)
user_five_mae_mean = find_mean(user_five_mae)
user_ten_mae_mean = find_mean(user_ten_mae)
user_twenty_mae_mean = find_mean(user_twenty_mae)
user_fifty_mae_mean = find_mean(user_fifty_mae)


item_five_rmse = item_five["test_rmse"]
item_ten_rmse = item_ten["test_rmse"]
item_twenty_rmse = item_twenty["test_rmse"]
item_fifty_rmse = item_fifty["test_rmse"]
item_five_mae = item_five["test_mae"]
item_ten_mae = item_ten["test_mae"]
item_twenty_mae = item_twenty["test_mae"]
item_fifty_mae = item_fifty["test_mae"]
item_five_rmse_mean = find_mean(item_five_rmse)
item_ten_rmse_mean = find_mean(item_ten_rmse)
item_twenty_rmse_mean = find_mean(item_twenty_rmse)
item_fifty_rmse_mean = find_mean(item_fifty_rmse)
item_fifty_mae_mean = find_mean(item_fifty_rmse)
item_five_mae_mean = find_mean(item_five_mae)
item_ten_mae_mean = find_mean(item_ten_mae)
item_twenty_mae_mean = find_mean(item_twenty_mae)
item_fifty_mae_mean = find_mean(item_fifty_mae)

print("Item five rmse mean")
print(item_five_rmse_mean)
print("Item Ten rmse mean")
print(item_ten_rmse_mean)
print("Item twenty rmse mean")
print(item_twenty_rmse_mean)
print("Item fifty rmse mean")
print(item_fifty_rmse_mean)
print("user five rmse mean")
print(user_five_rmse_mean)
print("user Ten rmse mean")
print(user_ten_rmse_mean)
print("User twenty rmse mean")
print(user_twenty_rmse_mean)
print("User fifty rmse mean")
print(item_fifty_rmse_mean)

user_cosine_rmse = user_cosine["test_rmse"]
user_cosine_mae = user_cosine["test_mae"]
user_pearson_rmse = user_pearson["test_rmse"]
user_pearson_mae = user_pearson["test_mae"]
user_msd_rmse = user_msd["test_rmse"]
user_msd_mae = user_msd["test_mae"]
user_cosine_rmse_mean = find_mean(user_cosine_rmse)
user_cosine_mae_mean = find_mean(user_cosine_mae)
user_pearson_rmse_mean = find_mean(user_pearson_rmse)
user_pearson_mae_mean = find_mean(user_pearson_mae)
user_msd_rmse_mean = find_mean(user_msd_rmse)
user_msd_mae_mean = find_mean(user_msd_mae)

item_cosine_rmse = item_cosine["test_rmse"]
item_cosine_mae = item_cosine["test_mae"]
item_pearson_rmse = item_pearson["test_rmse"]
item_pearson_mae = item_pearson["test_mae"]
item_msd_rmse = item_msd["test_rmse"]
item_msd_mae = item_msd["test_mae"]
item_cosine_rmse_mean = find_mean(item_cosine_rmse)
item_cosine_mae_mean = find_mean(item_cosine_mae)
item_pearson_rmse_mean = find_mean(item_pearson_rmse)
item_pearson_mae_mean = find_mean(item_pearson_mae)
item_msd_rmse_mean = find_mean(item_msd_rmse)
item_msd_mae_mean = find_mean(item_msd_mae)

print("user_cosine_rmse_mean")
print(user_cosine_rmse_mean)
print("user_cosine_mae_mean")
print(user_cosine_mae_mean)
print("user_pearson_rmse_mean")
print(user_pearson_rmse_mean)
print("user_pearson_mae_mean")
print(user_pearson_mae_mean)
print("user_msd_rmse_mean")
print(user_msd_rmse_mean)
print("user_msd_mae_mean")
print(user_msd_mae_mean)

print("item_cosine_rmse_mean")
print(item_cosine_rmse_mean)
print("item_cosine_mae_mean")
print(item_cosine_mae_mean)
print("item_pearson_rmse_mean")
print(item_pearson_rmse_mean)
print("item_pearson_mae_mean")
print(item_pearson_mae_mean)
print("item_msd_rmse_mean")
print(item_msd_rmse_mean)
print("item_msd_mae_mean")
print(item_msd_mae_mean)

print("Test")
print(test)
print(test["test_rmse"])
rmse = test["test_rmse"]
mean = find_mean(rmse)
print("Mean")
print(mean)
print(test["test_mae"])

print("The user array is ")
print(user)
print("The item array is")
print(item)
#for property, value in vars(test).items():
 #   print(property, ":", value)
