# %%
import pandas as pd
import numpy as np
import sklearn
import string 
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message



# %%
st.title('Movie Recommender')

st.write("""
### Not sure what to watch? 
Don't worry, we got you!
Just tell the our smart-bot your userId and you'll get a personal recommendation :)!
 
""")
# %%
links = pd.read_csv('links.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')



# %%
movie_ratings = pd.merge(movies, ratings, on="movieId")
rating_count_df = pd.DataFrame(movie_ratings.groupby('movieId')['rating'].count())
rating_count_df.rename(columns= {"rating" : "rating_count"})
rated = pd.merge(rating_count_df, movies, on= "movieId")
rated = pd.merge(rated, ratings, on= "movieId")
rating_mean = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
rated = pd.merge(rated, rating_mean, on= "movieId")
rated = rated.rename(columns= {"rating" : "overall_rating"})
rated = rated.drop(columns= ["rating_y", "timestamp"], axis = 1).rename(columns={"rating_x" : "rating_count"})
user_movies_df = pd.merge(rated, ratings, on= "movieId")
user_movies_df = user_movies_df.drop(columns=["userId_y", "timestamp"], axis=1).rename(columns={"userId_x" : "userId"})
user_movies = pd.pivot_table(data=user_movies_df, values='rating', index='userId', columns='movieId')


# %%
def movie_based(movie_id, n):
    m_ratings = user_movies[movie_id]
    similar_to_movies = user_movies.corrwith(m_ratings)
    corr_movies = pd.DataFrame(similar_to_movies, columns=['PearsonR'])
    corr_movies.dropna(inplace=True)
    movies_corr_summary = corr_movies.join(rated['rating_count'])
    movies_corr_summary.drop(movie_id, inplace=True) 
    top_movies = movies_corr_summary[movies_corr_summary['rating_count']>=30].sort_values('PearsonR', ascending=False).head(n)
    top_movies = top_movies.merge(movies, left_index=True, right_on="movieId")
    return top_movies["title"]

# %%
users_items = pd.pivot_table(data=user_movies_df, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')
users_items.fillna(0, inplace=True)
user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)

# %%
@st.cache
def user_based(user_id, n):
  weights = (user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]))
  no_user_rate = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
  weighted_averages = pd.DataFrame(no_user_rate.T.dot(weights), columns=["predicted_rating"])
  recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")
  top_recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(n)
  return top_recommendations["title"].tolist()


# %%

@st.cache(suppress_st_warning=True)
def chat_bot():
    message("Hi! I'm your personal recommender. Please tell me your userID.")
    user_id = st.text_input("UserId")
    recommendation = user_based(int((user_id)), 3)
    message("You will probably like these movies: ")
    st.write(recommendation)
chat_bot()

# %%
