#!/usr/bin/env python
# coding: utf-8

# In[156]:


import streamlit as st 
import pickle
import pandas as pd



# In[162]:


#links = pd.read_csv("/Users/hi/Desktop/data_science_wbs_school/Recommender_system/project/c1ba2d8cbaa22297e5d9b0b7a17fcb7awbsflix-dataset/ml-latest-small/links.csv")

#movies = pd.read_csv("/Users/hi/Desktop/data_science_wbs_school/Recommender_system/project/c1ba2d8cbaa22297e5d9b0b7a17fcb7awbsflix-dataset/ml-latest-small/movies.csv")
#ratings = pd.read_csv("/Users/hi/Desktop/data_science_wbs_school/Recommender_system/project/c1ba2d8cbaa22297e5d9b0b7a17fcb7awbsflix-dataset/ml-latest-small/ratings.csv")
#tags = pd.read_csv("/Users/hi/Desktop/data_science_wbs_school/Recommender_system/project/c1ba2d8cbaa22297e5d9b0b7a17fcb7awbsflix-dataset/ml-latest-small/tags.csv")


# In[163]:


movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies= pd.DataFrame(movies_dict)

rating_final = pickle.load(open('rating_final.pkl', 'rb'))
ratings = pickle.load(open('ratings.pkl', 'rb'))
rating_avg = pickle.load(open('rating_avg.pkl', 'rb'))
rating_count = pickle.load(open('rating_count.pkl', 'rb'))
users = pickle.load(open('users.pkl', 'rb'))
user_Id = pickle.load(open('user_Id.pkl', 'rb'))
movie_rating_tab = pickle.load(open('movie_rating_tab.pkl', 'rb'))
rating_final1 = pickle.load(open('rating_final1.pkl', 'rb'))


# In[180]:


st.set_page_config(page_title="WEBFLIX", layout="wide")
st.title('WBSFLIX')

#search_choices = ['Movie title', 'Genre', 'Rating', 'Tags']
#search_selected = st.sidebar.selectbox("Your choice please", search_choices)
movie_name = movies['title'].values



#dropdown_movie = st.selectbox('What would you like to watch today?', movie_name)

selected_movie_name= st.selectbox('What would you like to watch today?',movie_name)

st.write('You selected:', selected_movie_name)

def get_recommendations (movie_name,n=20):
    movie_rating_tab= pd.pivot_table(data=rating_final, values='rating', index='userId', columns='title')
      #movie_rating_tab = movie_rating_tab.fillna(0)
    rest_movie_ratings = movie_rating_tab[movie_name] 
    similar_to_rest = movie_rating_tab.corrwith(rest_movie_ratings)
    corr_rest = pd.DataFrame(similar_to_rest, columns = ['Pearson_coeif'])
    corr_rest.dropna(inplace=True)
    corr_rest.drop(movie_name, inplace=True)
    similar = corr_rest.merge(rating_final, on = "title")
    similar = similar.drop(['rating', 'timestamp', 'userId'], axis=1)
    similar = similar.drop_duplicates("title")
    similar_final= similar[similar['rating_count']>=100].sort_values(by = 'Pearson_coeif', ascending = False).head(n)
    similar_final = similar_final.drop(["overall_rating","rating_count","Pearson_coeif"], axis=1)
    similar_final["rating_avg"] = round(similar_final["rating_avg"], 1)
    similar_final["Year"]= similar_final["title"].str.findall("\d{4}")
    similar_final["title"]= similar_final["title"].str.replace('(\(\d{4})\)', '')
    similar_final = similar_final[['movieId', 'title', 'Year', 'genres', 'rating_avg']]
    similar_final_new= similar_final.set_index("movieId")
    return(similar_final_new)



with st.container():
    st.subheader('Top picks for you based on the movie you selected :')
    buffer1, col1, buffer2 = st.columns([2.5, 3, 1]) 
    is_clicked = col1.button(label = 'Recommended')
    if is_clicked :
        recommendations = get_recommendations(selected_movie_name)
        columns = st.columns((1, 4, 1))
        columns[1].write(recommendations)


        
        




def popular_table (movie_count):
  
  rating_avg = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
  rating_avg['overall_rating'] = round((ratings.groupby('movieId')['rating'].mean()))
  rating_count = pd.DataFrame(ratings.groupby('movieId')['rating'].count())
  rating_final  = rating_avg.merge(rating_count, on = 'movieId').merge(movies, on = "movieId")
  rating_final.rename(columns = {'rating_x':'rating_avg', 'rating_y':'rating_count'}, inplace = True)
  rating_final = rating_final[rating_final['rating_count']>=100].sort_values("rating_count", ascending=False)
  rating_final = rating_final.drop_duplicates(["title"])
  top_movies = rating_final.head(movie_count)


  return(top_movies)

# list of top 15 popular movies in our database
popular_movies = pickle.load(open('popular_all.pkl', 'rb'))


#session.options = st.multiselect(label="Select Movies", options=popular_movies)

#dataframe = None


#session.slider_count = st.slider(label="movie_count", min_value=5, max_value=138) 

#with st.container():
   # is_clicked = col1.button(label="Popular")
    #if is_clicked:
       # dataframe = popular_table(session.options, movie_count = session.slider_count, data= popular_movies) 
   # if dataframe is not None:
        #st.table(dataframe)
    
with st.container():
    st.subheader('Top 20:')
    buffer1, col1, buffer2 = st.columns([2.8, 3, 1]) 
    is_clicked = col1.button(label = "Popular")
    if is_clicked:
        columns = st.columns((0.5, 0.8, 0.3))
        columns[1].write(popular_movies)



def special_picks_for_you(user_id, n=30):
    user_rating_tab = movie_rating_tab.T.fillna(0)
  # computing cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarities = pd.DataFrame(cosine_similarity(user_rating_tab),
                                  columns=user_rating_tab.index, 
                                  index=user_rating_tab.index)
  # computing the weights for the users
    weights = (
        user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id])
            )
  # select movies that the inputed user has not rated, # dataframe for the non rated movies by all the user
    not_rated_movies = user_rating_tab.loc[user_rating_tab.index!=user_id, user_rating_tab.loc[user_id,:]==0]
  
    weighted_averages = pd.DataFrame(not_rated_movies.T.dot(weights), columns=["predicted_rating"])
    recommendations = weighted_averages.merge(movies, left_index=True, right_on="title").merge(rating_final1, on ="movieId").merge(users, on ="movieId").drop_duplicates("title")
    special_picks = recommendations[recommendations['rating_count']>100].sort_values("predicted_rating", ascending=False).head(n)
    special_picks = special_picks.drop(["userId","overall_rating","rating_count"], axis=1)
    special_picks["rating_avg"] = round(special_picks["rating_avg"], 1)
    special_picks["predicted_rating"] = round(special_picks["predicted_rating"], 1)

    special_picks["Year"]= special_picks["title"].str.findall("\d{4}")
    special_picks["title"]= special_picks["title"].str.replace('(\(\d{4})\)', '')
    special_picks_new = special_picks[['movieId', 'title', 'Year', 'genres', 'rating_avg', 'predicted_rating']]
    
    special_picks_new = special_picks_new.set_index('movieId')
    return(special_picks_new)
   
    
    
with st.container():
    st.subheader('Search by user Id:')
    user_id = user_Id
    user_choices =st.selectbox('See what other user have liked by User Id', user_id) 
    buffer1, col1, buffer2 = st.columns([3, 3, 1]) 
    is_clicked = col1.button(label = 'Show')
    if is_clicked :
        special_picks = special_picks_for_you(user_choices)
        columns = st.columns((1, 6.8, 1))
        columns[1].write(special_picks)

        


        
        


#import streamlit.components.v1 as components


# In[ ]:



                    


# In[ ]:





# In[ ]:





# In[ ]:




