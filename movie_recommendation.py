import pandas as pd 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv('toy_dataset.csv',index_col=0)
ratings.head()
ratings=ratings.fillna(0)
def standardize(row):
    new_row=(row-row.mean())/(row.max()-row.min())
    return new_row

ratings_std=ratings.apply(standardize)
ratings_std.head()

item_similarity=cosine_similarity(ratings_std.T)
item_similarity_df=pd.DataFrame(item_similarity,index=ratings.columns,columns=ratings.columns)
item_similarity_df.head()

#reccomending movies based on similarity score
def get_movies(movie_name,user_rating):
    similar_score=item_similarity_df[movie_name]*user_rating
    similar_score=similar_score.sort_values(ascending=False)
    
    return similar_score

print(get_movies("romantic3",1))


action_lover=[("action1",5),("romantic2",1),("romantic3",1)]

similar_movies=pd.DataFrame()
for movie,rating in action_lover:
    similar_movies=similar_movies.append(get_movies(movie,rating),ignore_index=True)
    
similar_movies.head()
similar_movies.sum().sort_values(ascending=False)




