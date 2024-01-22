import pandas as pd
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch

from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.cluster import KMeans

from torch.utils.data import Dataset
from surprise import Reader
from surprise import SVD

from time import time

#============================================================
# Reading CSV file, my own mini version of Yelp after
# preprocessing
#============================================================

sample = pd.read_csv('sample.csv')
yelp_user_filtered = pd.read_csv('yelp_user_filtered.csv')
yelp_business_filtered = pd.read_csv('yelp_business_filtered.csv')

############## Getting the dataset Ready ##############################

df = sample[['user_id','business_id','stars']]
df = df.rename(columns={'user_id':'userId','business_id':'businessId','stars':'rating'})
df = df[:28214]

###########################################################################
######## NON PERSONALIZED RECOMMENDER SYSTEM ##############################
###########################################################################


ratings_mean = df.groupby(['businessId'])[['rating']].mean().rename(columns = {'rating':'mean_rating'}).reset_index()
ratings_sum = df.groupby(['businessId'])[['rating']].sum().rename(columns = {'rating':'sum_rating'}).reset_index()
alpha = 1
ratings_sum['sum_rating_factor'] = ratings_sum['sum_rating'] + alpha*(df['rating'].mean())
ratings_count = df.groupby(['businessId'])[['rating']].count().rename(columns = {'rating':'count_rating'}).reset_index()
ratings_count['count_rating_factor'] = ratings_count['count_rating'] + alpha
ratings_damped = pd.merge(ratings_sum,ratings_count[['businessId','count_rating','count_rating_factor']], on=['businessId'], how='left')
ratings_damped['damped_mean'] = ratings_damped['sum_rating_factor']/ratings_damped['count_rating_factor']
rating_mean_dampmean = pd.merge(ratings_mean[['businessId','mean_rating']],ratings_damped[['businessId','damped_mean']], on =['businessId'], how= 'left')
rating_mean_dampmean = rating_mean_dampmean.sort_values(['mean_rating'], ascending=False)

def non_personalised():
    num_of_recommendations = 10

    list_of_predictions = rating_mean_dampmean['businessId'][:num_of_recommendations].tolist()
    business_ = []
    for i in range(len(list_of_predictions)):
            business_name_ = list_of_predictions[i]
            business_name = yelp_business_filtered['name'].loc[yelp_business_filtered['business_id'] == business_name_]
            business_.append(business_name.tolist()[0])
    for i in range(len(business_)):
            print("Your number",i+1,"recommendation is",business_[i])
    

###########################################################################
######## COLD START RECOMMENDER SYSTEM ####################################
###########################################################################


# This is solved by using K-Means Clustering Algorithm 
#=========================================================================
m_clus = int((1/4)*len(yelp_business_filtered['city'].unique()))
kmeans_city = KMeans(n_clusters=m_clus,init='k-means++')
kmeans_city.fit(yelp_business_filtered[['latitude','longitude']].values)
x = kmeans_city.labels_
yelp_business_filtered['city_category'] = x
#=========================================================================

# function that outputs the recommendations for the new user
# this is based on location, if there aren't around recommendations
# the algorithm will branch out to the next function increase radius function
# to suggest more restaurants 
def new_user():
    #=================================================================================================
    # gets input from the user 
    #=================================================================================================
    longitude = float(input("Your Longitude, ex:-75.2: "))
    latitude = float(input("Your Latitude, ex:39.9: "))
    num_of_recommendations = 10
    cluster = kmeans_city.predict(np.array([longitude,latitude]).reshape(1,-1))[0]
    #=================================================================================================
    # getting the number of number of restaurants for that city cluster 
    #=================================================================================================
    restaurants = yelp_business_filtered[yelp_business_filtered['city_category']==cluster]
    n_o_restaurants = restaurants.shape[0]
    list_of_new_id = restaurants['business_id'].values.tolist()

    if num_of_recommendations>n_o_restaurants:
        #=============================================================================================
        # You can tweak the radius a little to give the number of recommendations requested by user
        #=============================================================================================
        list_of_new_id = increase_radius(int(latitude),int(longitude),num_of_recommendations)
        
    y = yelp_business_filtered[yelp_business_filtered['business_id'].isin(list_of_new_id)] #.values.tolist()
    y = y[:num_of_recommendations]
    business_name_ = y['business_id'].tolist()
    business_ = []
    for i in range(len(business_name_)):
        business_name = yelp_business_filtered['name'].loc[yelp_business_filtered['business_id'] == business_name_[i]]
        business_.append(business_name.tolist()[0])
    return business_

# to branch out more restaurants if the user wants a large number
# of recmmendations
def increase_radius(lat,long,number):
    business_id = []
    n = 0.01
    while len(business_id) <= number:
        
        #==================================================
        # lat values
        lat_values = [lat + n, lat - n]
        #==================================================
        # long values
        long_values = [long + n, long - n]
        #==================================================
        # for loop to check the regions
        for lat in lat_values:
            for long in long_values:
                #==================================================
                # get the region of cluster
                cluster = kmeans_city.predict(np.array([long,lat]).reshape(1,-1))[0]
                #==================================================
                # getting the number of number of restaurants for that city cluster
                restaurants = yelp_business_filtered[yelp_business_filtered['city_category']==cluster]
                #==================================================
                # number of restaurants in that region
                n_o_restaurants = restaurants.shape[0]

                list_of_new_id = restaurants['business_id'].values.tolist()
                res = [x for x in list_of_new_id + business_id if x not in business_id]
                business_id_with_more_star = []
                for i in res:
                    star_now = yelp_business_filtered['stars'][yelp_business_filtered['business_id']==i]
                    if int(star_now) >= 4:
                        business_id_with_more_star.append(i)
                if len(res) == 0:
                    pass
                else:
                    for i in range(len(business_id_with_more_star)):
                        business_id.append(business_id_with_more_star[i])
        n = n + 0.01
        
    return business_id



###########################################################################
######## PERSONALIZED RECOMMENDER SYSTEM ##################################
###########################################################################

from surprise import Dataset
from surprise import Reader

# =========== Getting user item rating dataframe ===============
user_item_rating = sample.loc[:, ['user_id', 'business_id','super_stars','date']]
user_item_rating = user_item_rating.reset_index(drop = True)

# =========== Train test split =================================

user_item_rating["date"] = pd.to_datetime(user_item_rating["date"])
user_item_rating["date"] = user_item_rating["date"].astype('datetime64[ns]')
user_item_rating.sort_values(by='date', inplace=False)

train, test_val = train_test_split(user_item_rating, test_size=0.4,random_state=None, shuffle=False, stratify=None)
val, test = train_test_split(test_val, test_size=0.5,random_state=None, shuffle=False, stratify=None)


reader = Reader(rating_scale = (0.0, 5.0))

trainset = train.loc[:,['user_id', 'business_id', 'super_stars']]
trainset.columns = ['userID', 'itemID','rating']

valset = val.loc[:,['user_id', 'business_id', 'super_stars']]
valset.columns = ['userID', 'itemID','rating']

testset = test.loc[:,['user_id', 'business_id', 'super_stars']]
testset.columns = ['userID', 'itemID','rating']

train_data = Dataset.load_from_df(trainset[['userID', 'itemID','rating']], reader)
val_data = Dataset.load_from_df(valset[['userID', 'itemID','rating']], reader)
test_data = Dataset.load_from_df(testset[['userID', 'itemID','rating']], reader)

train_sr = train_data.build_full_trainset()
val_sr_before = val_data.build_full_trainset()
val_sr = val_sr_before.build_testset()
test_sr_before = test_data.build_full_trainset()
test_sr = test_sr_before.build_testset()

#######################################################################
#################### SVD  #############################################
#######################################################################
user_item_rating = sample[["user_id", 'business_id', 'date', 'stars']]

user_item_rating = user_item_rating.loc[:,['user_id', 'business_id', 'stars']]

user_item_rating.columns = ['userID', 'itemID','rating']

user_item_ratings = Dataset.load_from_df(user_item_rating[['userID', 'itemID','rating']], reader)

user_item_ratings = user_item_ratings.build_full_trainset()

#===================================================================================================
# training SVD algorithm in the optimal parameter 
#====================================================================================================

algo_real = SVD(n_epochs = 40, lr_all = 0.005, reg_all = 0.05)

algo_real.fit(user_item_ratings)

def SVD_recommend():
    print("There are",user_item_rating.userID.nunique(),"number of users and their ids range between 0 and",user_item_rating.userID.nunique()-1) 
    print('Recommendation for user_id:')
    user = int(input())
    while user >= user_item_rating.userID.nunique():
        print("There are no such user")
        print("Please enter a valid Id")
        user = int(input())
    num_of_recommendations = 10
    list_of_predictions = []
    unique_business = sample.business_id.unique()
    business_ = []

    for i in range(sample.business_id.nunique()):
            business_id = unique_business[i]
            y_hat = algo_real.predict(user, business_id).est
            list_of_predictions.append([business_id,y_hat])

    list_of_predictions.sort(key=lambda x: x[1],reverse = True)
    list_of_predictions = list_of_predictions[:num_of_recommendations]
    for i in range(len(list_of_predictions)):
            business_name_ = list_of_predictions[i]
            business_name = yelp_business_filtered['name'].loc[yelp_business_filtered['business_id'] == business_name_[0]]
            business_.append(business_name.tolist()[0])

    for i in range(len(business_)):
            print("Your number",i+1,"recommendation is",business_[i])

#######################################################################
#################### Deep Model Defining ##############################
#######################################################################

class RecSysModel(nn.Module):
    def __init__(self, n_users, n_business):
        super().__init__()
        # trainable lookup matrix for shallow embedding vectors
    
        self.user_embed = nn.Embedding(n_users, 32)
        self.business_embed = nn.Embedding(n_business,32)
        self.out = nn.Linear(64,1)
        
    def forward(self, users, business, rating=None):
        user_embeds = self.user_embed(users)
        business_embeds = self.business_embed(business)
        output = torch.cat([user_embeds,business_embeds], dim=1)
        
        output = self.out(output)
        
        return output

#====================================================================
# label encoder is a bit like getting all the uniques values
# to make sure each userid fits into the total unique user Id
#====================================================================

lbl_user = preprocessing.LabelEncoder()
lbl_business = preprocessing.LabelEncoder()

#====================================================================
# fit_transform() is used on the training data so that we can scale 
# the training data and also learn the scaling parameters of that data.
#====================================================================

df.userId = lbl_user.fit_transform(df.userId.values)
df.businessId = lbl_business.fit_transform(df.businessId.values)

#====================================================================
# label encoder is a bit like getting all the uniques values
# to make sure each userid fits into the total unique user Id
#====================================================================

lbl_user = preprocessing.LabelEncoder()
lbl_business = preprocessing.LabelEncoder()

#====================================================================
# fit_transform() is used on the training data so that we can scale 
# the training data and also learn the scaling parameters of that data.
#====================================================================

df.userId = lbl_user.fit_transform(df.userId.values)
df.businessId = lbl_business.fit_transform(df.businessId.values)

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

model = RecSysModel(
    n_users = len(lbl_user.classes_),
    n_business = len(lbl_business.classes_),
).to(device)

#====================================================================
# Model's optimizer
#====================================================================

optimizer = torch.optim.Adam(model.parameters())
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size =3, gamma=0.7)

#=====================================================================
# Model Weight initializer
#=====================================================================

PATH = "model_weights.pth"

model.load_state_dict(torch.load(PATH))
model.eval()

#=====================================================================
# To make predictions
#=====================================================================
def predict_best_restaurants():
    print("There are",df.userId.nunique(),"number of users and their ids range between 0 and",df.userId.nunique()-1) 
    print('Recommendation for user_id:')
    user = int(input())

    while user >= df.userId.nunique():
        print("There are no such user")
        print("Please enter a valid Id")
        user = int(input())
    num_of_rec = 10
    user = torch.tensor([user], dtype=torch.long)

    list_of_predictions = []
    business_ = []
    unique_business = df.businessId.unique().tolist()
    for i in range(df.businessId.nunique()):
        business_id = unique_business[i]
        business = torch.tensor([business_id], dtype=torch.long)
        y_hat = model(user,business)
        y_hat = y_hat.detach().numpy()
        list_of_predictions.append([business_id,y_hat[0]])
        
    list_of_predictions.sort(key=lambda x: x[1],reverse = True)
    list_of_predictions = list_of_predictions[:num_of_rec]

    for i in range(len(list_of_predictions)):
        business_name_ = list_of_predictions[i][0]
        business_name = yelp_business_filtered['name'].loc[yelp_business_filtered['business_id'] == business_name_]
        business_.append(business_name.tolist()[0])
        
    for i in range(len(business_)):
        print("Your number",i+1,"recommendation is",business_[i])

#########################################################################
#################### INTERACTIVE INTERFACE ##############################
#########################################################################

continue_ = 'yes'

while continue_ == 'yes':
    print("Welcome to the resturant recommender \r")
    print("Would you like a personalised or non personalised recommendation? Type 'personalised' or 'non personalised'")
    type_of_recommendation = str(input())

    while not (type_of_recommendation == 'personalised' or type_of_recommendation == 'non personalised'):
                print("Sorry, can you type 'personalised' or 'non personalised'")
                type_of_recommendation = input()

    if type_of_recommendation == 'personalised':
        print("Welcome to the personalised resturant recommender \r")
        print("Are you a new or existing user? Type 'new' or 'existing' \r")
        user_input = input()

        while not (user_input == 'new' or user_input == 'existing'):
                print("Sorry, can you type 'new' or 'existing'")
                user_input = input()

        if user_input == 'new':
            #===============================================================================================
            # Listing all the recommendations for the new user
            #===============================================================================================
            print("This is your recommendation by location. Only restaurants with 4 ratings and above will be recommended")
            print("What is your location?")
            ########################################################################################
            t_non_personalised_1 = time()
            
            list_of_recommendation_for_new_user = new_user()

            t_non_personalised_2 = time()

            #print ('New user clustering algo %f' %(t_non_personalised_2-t_non_personalised_1))
            ########################################################################################

            for i in range(len(list_of_recommendation_for_new_user)):
                print("Your number",i+1,"recommendation is",list_of_recommendation_for_new_user[i])
            print("Would you want to start again? yes/no :")
            continue_ = input()
        else:
            print("This is your recommendation by looking at users who liked the same things as you did.")
            print("Would you like a more accurate recommendations or a little adventurous recommendations? Type 'accurate' or 'adventurous'\r")
            user_input = input()

            while not (user_input == 'accurate' or user_input == 'adventurous'):
                print("This is your recommendation is based on what you have liked in the past.")
                print("Sorry, can you type 'adventurous' or 'accurate'")
                user_input = input()

            if user_input == 'accurate':
                ########################################################################################
                t_non_personalised_1 = time()
                
                predict_best_restaurants()

                t_non_personalised_2 = time()

                #print ('Personalised Deep Learning algorithm takes %f' %(t_non_personalised_2-t_non_personalised_1))
                ########################################################################################

            else:

                print("This is your recommendation is recommends restaurants that you might be a littlle different from what you are normally used to.")

                ########################################################################################
                t_non_personalised_1 = time()
                
                SVD_recommend()

                t_non_personalised_2 = time()

                #print ('Personalised SVD algorithm takes %f' %(t_non_personalised_2-t_non_personalised_1))
                ########################################################################################
                
            print("Would you want to start again? yes/no :")
            continue_ = input()
    else:
        print("Welcome to the non personalised resturant recommender \r")
        print("This is your recommendation is recommends restaurants that are famous and highy rated")
        ########################################################################################
        t_non_personalised_1 = time()
                
        non_personalised()

        t_non_personalised_2 = time()

        #print ('Non personalised algorithm takes %f' %(t_non_personalised_2-t_non_personalised_1))
        ########################################################################################
        print("Would you want to start again? yes/no :")
        continue_ = input()

print("Thank you and hope you enjoy your meal :)")