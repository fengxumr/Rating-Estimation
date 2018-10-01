import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import datasets, metrics
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from surprise import SVDpp, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split, cross_validate
from xgboost import XGBRegressor
from xgboost import plot_importance
from matplotlib import pyplot

## load all data
book_info = pd.read_csv('Books information.csv')
genre_map = pd.read_csv('Genres Mapping.csv')
user_data = pd.read_csv('User Data.csv')
words_in_book = pd.read_csv('Words in Books Data.csv')
words_map = pd.read_csv('Words Mapping.csv')


## train dataset
train_df = user_data[['User ID', 'User Read Books (2017)', 'Average Rating (2017)']]
train_ar = [[a[0], int(b), float(a[2])] for a in train_df.values for b in a[1].split(', ')]
user_book_rate = pd.DataFrame(train_ar, columns=train_df.columns)

## predict dataset
test_df = user_data[['User ID', 'User Read Books (2018)']]
test_ar = [[a[0], int(b)] for a in test_df.values for b in a[1].split(', ')]


## first model and training
reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(user_book_rate, reader)
trainset = data.build_full_trainset()
# algo = SVDpp(n_factors=2,n_epochs=75,lr_all=0.05,reg_all=0.1)
algo = SVDpp(n_factors=3,n_epochs=300,lr_all=0.01,reg_all=0.2)
algo.fit(trainset)
user_latent = algo.pu
book_latent = algo.qi


## final result of first model
final_df = user_data[['User ID', 'User Read Books (2017)', 'User Read Books (2018)', 'Average Rating (2017)']]
final_ar = [[a[0], (len(a[1].split(', ')) * float(a[3]) + sum([algo.predict(a[0], int(b)).est for b in a[2].split(', ')])) / (len(a[1].split(', ')) + len(a[2].split(', ')))] for a in final_df.values]


## user difficulty embedding
user_diff_ar = user_data['User Difficulty Choice'].values
mlb = MultiLabelBinarizer(classes = [1,2,3,4,5])
user_diff_code = mlb.fit_transform([([int(a)]) if a in '12345' else (1,2,3,4,5) for a in user_diff_ar])
dic_user_diff = dict(zip(user_data['User ID'].values, user_diff_code))

## book difficulty embedding
book_diff_ar = book_info['Difficulty (Reader suggested)'].values
mlb = MultiLabelBinarizer(classes = [1,2,3,4,5])
book_diff_code = mlb.fit_transform([([int(a)]) for a in book_diff_ar])
dic_book_diff = dict(zip(book_info['Book ID'].values, book_diff_code))

## book genre embedding
book_genre_ar = book_info['Book Genre'].values
mlb = LabelBinarizer()
book_genre_code = mlb.fit_transform(book_genre_ar)
dic_book_genre = dict(zip(book_info['Book ID'].values, book_genre_code))

## book most sell places embedding
book_store_ar = book_info['Most Sold At'].values
mlb = LabelBinarizer()
book_store_code = mlb.fit_transform(book_store_ar)
dic_book_store = dict(zip(book_info['Book ID'].values, book_store_code))

## book words embedding
book_words_ar = words_in_book['Words in Book'].values
mlb = MultiLabelBinarizer()
book_words_code = mlb.fit_transform([a.split('|') for a in book_words_ar])
dic_book_words = dict(zip(words_in_book['Book ID'].values, book_words_code))


## second model and training
# X_train = [np.concatenate((user_latent[algo.trainset.to_inner_uid(a[0])], book_latent[algo.trainset.to_inner_iid(a[1])], dic_user_diff[a[0]], dic_book_diff[a[1]], dic_book_genre[a[1]], dic_book_store[a[1]])) for a in user_book_rate.values]
X_train = [np.concatenate((dic_user_diff[a[0]], dic_book_diff[a[1]], dic_book_genre[a[1]], dic_book_store[a[1]])) for a in user_book_rate.values]
# X_train = pd.DataFrame(X_train, columns=['user_id_'+str(a) for a in range(3)] + ['book_id_'+str(a) for a in range(3)] +  ['dic_user_diff_'+str(a) for a in range(1,6)] + ['dic_book_diff_'+str(a) for a in range(1,6)] + ['dic_book_genre_'+str(a) for a in range(31)] + ['store_'+str(a) for a in range(1,6)])
y_train = [a[2]-algo.predict(a[0], a[1]).est for a in user_book_rate.values]
model = XGBRegressor()
model.fit(X_train, y_train)
# plot_importance(model)
# pyplot.show()


## self check
y_pred = model.predict(X_train)
accuracy = metrics.r2_score(y_train, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


## predict of second model
# X_test = [np.concatenate((user_latent[algo.trainset.to_inner_uid(a[0])], book_latent[algo.trainset.to_inner_iid(a[1])], dic_user_diff[a[0]], dic_book_diff[a[1]], dic_book_genre[a[1]], dic_book_store[a[1]])) for a in test_ar]
X_test = [np.concatenate((dic_user_diff[a[0]], dic_book_diff[a[1]], dic_book_genre[a[1]], dic_book_store[a[1]])) for a in test_ar]
y_pred_new = model.predict(X_test)
dic_con = defaultdict(dict)
for i in range(len(test_ar)):
    dic_con[test_ar[i][0]][test_ar[i][1]] = y_pred_new[i]


## final result of first model
final_df_2 = user_data[['User ID', 'User Read Books (2017)', 'User Read Books (2018)', 'Average Rating (2017)']]
final_ar_2 = [[a[0], (len(a[1].split(', ')) * float(a[3]) + sum([algo.predict(a[0], int(b)).est+dic_con[a[0]][int(b)] for b in a[2].split(', ')])) / (len(a[1].split(', ')) + len(a[2].split(', ')))] for a in final_df_2.values]
rel = pd.DataFrame(final_ar_2, columns=['User ID', 'Average Rating (2018)'])


## save to file
rel.to_csv('out.csv', index=False)





# final_df_2 = user_data[['User ID', 'User Read Books (2018)', 'User Difficulty Choice']]
# final_ar_2 = [[a[0], int(b), algo.predict(a[0], int(b)).est+dic_con[a[0]][int(b)], a[2]] for a in final_df_2.values for b in a[1].split(', ')]


# train_df = user_data[['User ID', 'User ID', 'Rating', 'User Difficulty Choice']]
# train_ar = [[a[0], int(b), float(a[2]), a[3]] for a in train_df.values for b in a[1].split(', ')]
# user_book_rate = pd.DataFrame(train_ar, columns=train_df.columns)







