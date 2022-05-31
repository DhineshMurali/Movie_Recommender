from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def home(request):
    return render(request,"home.html")

def predict(request):
    return render(request,"predict.html")

def result(request):
    '''
    data = pd.read_csv('D:/ML Projects/Django + ML prediction/Engineering_graduate_salary.csv')
    # Degree
    data['Degree'] = data['Degree'].map({'B.Tech/B.E.': 1, 'M.Tech./M.E.': 2, 'MCA': 3, 'M.Sc. (Tech.)': 4}).astype(int)
    # Gender
    data['Gender'] = data['Gender'].map({'m': 0, 'f': 1}).astype(int)
    # Train Test Split

    X = pd.DataFrame(np.c_[data['Gender'], data['10percentage'], data['12percentage'], data['CollegeTier'], data[
        'Degree'], data['collegeGPA'], data['GraduationYear'], data['Domain'], data['agreeableness'], data[
                               'openess_to_experience']],
                     columns=['Gender', '10percentage', '12percentage', 'CollegeTier', 'Degree', 'collegeGPA',
                              'GraduationYear', 'Domain', 'agreeableness', 'openess_to_experience'])
    Y = pd.DataFrame(data['Salary'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=30)

    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    var6 = float(request.GET['n6'])
    var7 = float(request.GET['n7'])
    var8 = float(request.GET['n8'])
    var9 = float(request.GET['n9'])
    var10 = float(request.GET['n10'])

    pred = model.predict(np.array([var1,var2,var3,var4,var5,var6,var7,var8,var9,var10]).reshape(1,-1))
    #pred = round(pred[0])

    salary = "The predicted salary is $"+str(pred)
    '''
    movies_df = pd.read_csv('D:/ML Projects/Movie-recommendation-system-main/movies.csv', usecols=['movieId', 'title'], dtype={'movieId': 'int32', 'title': 'str'})
    rating_df = pd.read_csv('D:/ML Projects/Movie-recommendation-system-main/ratings.csv', usecols=['userId', 'movieId', 'rating'],
                            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

    df = pd.merge(rating_df, movies_df, on='movieId')
    combine_movie_rating = df.dropna(axis=0, subset=['title'])
    movie_ratingCount = (combine_movie_rating.
        groupby(by=['title'])['rating'].
        count().
        reset_index().
        rename(columns={'rating': 'totalRatingCount'})
    [['title', 'totalRatingCount']]
        )
    rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title',
                                                              how='left')

    popularity_threshold = 50
    rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

    movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)

    movie_features_df_matrix = csr_matrix(movie_features_df.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(movie_features_df_matrix)

    #query_index = np.random.choice(movie_features_df.shape[0])
    query_index = int(request.GET['n1'])
    print(query_index)
    distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1),n_neighbors=10)
    pred =[]
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]],distances.flatten()[i]))
            pred.append(movie_features_df.index[indices.flatten()[i]])

    #pred = model.predict(np.array([var1, var2, var3, var4, var5, var6, var7, var8, var9, var10]).reshape(1, -1))
    # pred = round(pred[0])

    #mov = "The recommended movies are $" + pred[0]
    print(pred)
    return render(request,"predict.html",{"result2":pred,"mov_name":movie_features_df.index[query_index]})