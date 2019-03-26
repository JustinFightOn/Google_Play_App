import os
import math
import pandas as pd
import numpy as np
from joblib import load

#default_path = r'C:\Users\Cyndi\Documents\Git Repository\Stat 404 Private\Project_Final'
#os.chdir(default_path)

# Load model
model = load('app_installations.joblib')

# User input data
def input_data():
    INPUT = pd.DataFrame()
    i = 0
    while True:
        INPUT.loc[i,'Category'] = input("Please enter category: ")
        INPUT.loc[i,'Rating'] = input("Please enter Rating: ")
        INPUT.loc[i,'Size'] = input("Please enter Size: ")
        INPUT.loc[i,'Type'] = input("Please enter Type: ")
        INPUT.loc[i,'Price'] = input("Please enter Price: ")
        INPUT.loc[i,'Content Rating'] = input("Please enter Content Rating: ")
        INPUT.loc[i,'Genres'] = input("Please enter Genres: ")
        INPUT.loc[i,'Last Updated'] = input("Please enter Last Updated: ")
        INPUT.loc[i,'Android Ver'] = input("Please enter Android Ver: ")
        if input("More inputs? Yes = Continue, No = See Results: ") == 'No':
            break
        i += 1
    return INPUT

# Data processing:
def data_processing(DF):
    DF['Rating'] = DF['Rating'].apply(lambda x: float(x))

    DF['Size'] = DF['Size'].apply(lambda x: x.replace('M', '000000').replace('.', '').replace('k', '000'))
    DF['size_category'] = ' '
    for i in range(len(DF)):
        if DF.loc[i, 'Size'] == 'Varies with device':
            DF.loc[i, 'size_category'] = 'Varies with device'
        elif  int(DF.loc[i, 'Size']) < 10000000:
            DF.loc[i, 'size_category'] = 'Size < 10MB'
        elif  int(DF.loc[i, 'Size']) < 50000000:
            DF.loc[i, 'size_category'] = '10MB<=Size<50MB'
        else:
            DF.loc[i, 'size_category'] = 'Size >= 50MB'
        
    def log_transform(x):
        return math.log10(x+1)

    def float_price(x):
        return float(x.replace('$', ''))
    
    DF['Price'] = DF['Price'].apply(float_price)
    DF['log_Price'] = DF['Price'].apply(log_transform)
    
    DF['Content Rating'] = ['Mature 17+' if r == 'Adults only 18+' else 'Everyone' if r == 'Unrated' else r for r in DF['Content Rating']]
    
    DF['Last Updated'] = pd.to_datetime(DF['Last Updated'])
    DF['Year'] = [d.year for d in DF['Last Updated']]
    DF['Month'] = [d.month for d in DF['Last Updated']]
    
    DF['Android Version Group'] = ' '
    for i in range(len(DF)):
        if DF.loc[i, 'Android Ver'] == 'Varies with device':
            DF.loc[i, 'Android Version Group'] = 'Varies with device'
        elif  int(DF.loc[i, 'Android Ver'][0]) < 4:
            DF.loc[i, 'Android Version Group'] = 'Allow Older than Android Ver 4'
        elif  int(DF.loc[i, 'Android Ver'][0]) < 5:
            DF.loc[i, 'Android Version Group'] = 'Android Version 4 and up'
        else:
            DF.loc[i, 'Android Version Group'] = 'Android Version newer than 5'
    
    DF['Genres_count'] = DF['Genres'].apply(lambda x: len(x.split(';')))
    DF['First_Genres'] = DF['Genres'].str.split(';', expand=True)[0]

def add_dummies(DF):
    DF = DF[['Category', 'Type', 'Content Rating', 'Android Version Group', 'size_category', \
           'log_Price', 'Year', 'Month', 'Genres_count', 'First_Genres', 'Rating']]
    DF = pd.get_dummies(DF)
    INDEX = ['log_Price', 'Year', 'Month', 'Genres_count', 'Category_ART_AND_DESIGN',
       'Category_AUTO_AND_VEHICLES', 'Category_BEAUTY',
       'Category_BOOKS_AND_REFERENCE', 'Category_BUSINESS', 'Category_COMICS',
       'Category_COMMUNICATION', 'Category_DATING', 'Category_EDUCATION',
       'Category_ENTERTAINMENT', 'Category_EVENTS', 'Category_FAMILY',
       'Category_FINANCE', 'Category_FOOD_AND_DRINK', 'Category_GAME',
       'Category_HEALTH_AND_FITNESS', 'Category_HOUSE_AND_HOME',
       'Category_LIBRARIES_AND_DEMO', 'Category_LIFESTYLE',
       'Category_MAPS_AND_NAVIGATION', 'Category_MEDICAL',
       'Category_NEWS_AND_MAGAZINES', 'Category_PARENTING',
       'Category_PERSONALIZATION', 'Category_PHOTOGRAPHY',
       'Category_PRODUCTIVITY', 'Category_SHOPPING', 'Category_SOCIAL',
       'Category_SPORTS', 'Category_TOOLS', 'Category_TRAVEL_AND_LOCAL',
       'Category_VIDEO_PLAYERS', 'Category_WEATHER', 'Type_Free', 'Type_Paid',
       'Content Rating_Everyone', 'Content Rating_Everyone 10+',
       'Content Rating_Mature 17+', 'Content Rating_Teen',
       'Android Version Group_Allow Older than Android Ver 4',
       'Android Version Group_Android Version 4 and up',
       'Android Version Group_Android Version newer than 5',
       'Android Version Group_Varies with device',
       'size_category_10MB<=Size<50MB', 'size_category_Size < 10MB',
       'size_category_Size >= 50MB', 'size_category_Varies with device',
       'First_Genres_Action', 'First_Genres_Adventure', 'First_Genres_Arcade',
       'First_Genres_Art & Design', 'First_Genres_Auto & Vehicles',
       'First_Genres_Beauty', 'First_Genres_Board',
       'First_Genres_Books & Reference', 'First_Genres_Business',
       'First_Genres_Card', 'First_Genres_Casino', 'First_Genres_Casual',
       'First_Genres_Comics', 'First_Genres_Communication',
       'First_Genres_Dating', 'First_Genres_Education',
       'First_Genres_Educational', 'First_Genres_Entertainment',
       'First_Genres_Events', 'First_Genres_Finance',
       'First_Genres_Food & Drink', 'First_Genres_Health & Fitness',
       'First_Genres_House & Home', 'First_Genres_Libraries & Demo',
       'First_Genres_Lifestyle', 'First_Genres_Maps & Navigation',
       'First_Genres_Medical', 'First_Genres_Music',
       'First_Genres_Music & Audio', 'First_Genres_News & Magazines',
       'First_Genres_Parenting', 'First_Genres_Personalization',
       'First_Genres_Photography', 'First_Genres_Productivity',
       'First_Genres_Puzzle', 'First_Genres_Racing',
       'First_Genres_Role Playing', 'First_Genres_Shopping',
       'First_Genres_Simulation', 'First_Genres_Social', 'First_Genres_Sports',
       'First_Genres_Strategy', 'First_Genres_Tools',
       'First_Genres_Travel & Local', 'First_Genres_Trivia',
       'First_Genres_Video Players & Editors', 'First_Genres_Weather',
       'First_Genres_Word', 'Rating']
    DF = DF.reindex(columns=INDEX, fill_value=0)
    return DF
    
# Final predictions
def prediction(DF):
    prediction = np.round(10**model.predict(DF))
    for i in range(len(prediction)):
        print(f'The predicted number of installations for the input data is: {prediction[i]}')
    
#Sample input:    
#ART_AND_DESIGN	4.1	19M		Free	0	Everyone	Art & Design	7-Jan-18	4.0.3 and up
#ART_AND_DESIGN	3.9	14M		Free	0	Everyone	Art & Design;Pretend Play	15-Jan-18	4.0.3 and up
#ART_AND_DESIGN	4.7	8.7M	Free	0	Everyone	Art & Design	1-Aug-18	4.0.3 and up

# Run the models
def run_model():
    DF = input_data()
    data_processing(DF)
    DF = add_dummies(DF)
    prediction(DF)

run_model()
