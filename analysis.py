import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures    

def main():
    #set up seaborn 
    sns.set_theme()

    # Read in the Movies dataset
    print("Importing movies.csv... \n")
    df = pd.read_csv("movies.csv")

    #preprocess data
    df = preprocess(df)

    print("Printing Dataset Statistics...")
    print(df.describe(), "\n")

    analyse_languages(df)

def preprocess(df):
    print("Pre-processing data... \n")

    #drop NaN runtime values
    df = df.dropna(subset=["runtime"])

    #create new column for profit
    df["profit"] = df["revenue"] - df["budget"]

    #create new column for total vote score
    df["total_vote_score"] = df["vote_average"] * df["vote_count"]

    #create new column for season of release
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["season"] = df["release_date"].apply(get_season)

    return df

def filter_movies(df):
    # filter out movies that have negative profit
    # df = df[df["profit"] > 0]

    # filter out movies whose popularity isn't in the top 75%
    df = df[df["popularity"] > df["popularity"].quantile(0.75)]

    # print the value of the 75th percentile for popularity
    print("75th Percentile for Popularity: ", df["popularity"].quantile(0.75), "\n")

    return df

def analyse_languages(df):
    # filter movies
    df = filter_movies(df)

    pprint.pprint(df)

    #create dictionary to store genre and occurence count
    language_dict = {}

    # use eval to convert string to list of dictionaries
    for index,row in df.iterrows():
        languages = eval(row["spoken_languages"])

        # loop through each genre in the list and update the dictionary
        for language in languages:
            language_name = iso_to_language(language["iso_639_1"])
            
            if language_name in language_dict:
                language_dict[language_name] += 1
            else:
                language_dict[language_name] = 1    

    # get the top 40 keywords in ascending order
    language_dict = {k: v for k, v in sorted(language_dict.items(), key=lambda item: item[1], reverse=True)[:30]}

    #sort dictionary by value and print
    language_dict = {k: v for k, v in sorted(language_dict.items(), key=lambda item: item[1], reverse=True)}

    pprint.pprint(language_dict)
    plot_language_count(language_dict)

def plot_language_count(language_dict):
    #plot genre vs count as horizontal bar chart
    plt.barh(list(language_dict.keys()), language_dict.values(), color='b')
    plt.xlabel("Number of Popular Movies")
    plt.ylabel("Spoken Languages")
    plt.title("Number of Popular Movies per Spoken Language")
    plt.show()

def iso_to_language(iso_code):
    languages = {
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'ru': 'Russian',
        'it': 'Italian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'pt': 'Portuguese',
        'la': 'Latin',
        'th': 'Thai',
        'cn': 'Chinese',
        'hi': 'Hindi',
        'pl': 'Polish',
        'ko': 'Korean',
        'he': 'Hebrew',
        'cs': 'Czech',
        'el': 'Greek',
        'hu': 'Hungarian',
        'sv': 'Swedish',
        'tr': 'Turkish',
        'ur': 'Urdu',
        'vi': 'Vietnamese',
        'ro': 'Romanian',
        'no': 'Norwegian',
        'uk': 'Ukrainian',
        'fa': 'Persian',
        'yi': 'Yiddish',
        'da': 'Danish'
    }
    return languages.get(iso_code, 'Unknown language')

def analyse_genre(df):
    # filter movies
    df = filter_movies(df)

    pprint.pprint(df)

    #create dictionary to store genre and occurence count
    genre_dict = {}

    # use eval to convert string to list of dictionaries
    for index,row in df.iterrows():
        genres = eval(row["genres"])

        # loop through each genre in the list and update the dictionary
        for genre in genres:
            genre_name = genre["name"]
            
            if genre_name in genre_dict:
                genre_dict[genre_name] += 1
            else:
                genre_dict[genre_name] = 1

    #sort dictionary by value and print
    genre_dict = {k: v for k, v in sorted(genre_dict.items(), key=lambda item: item[1], reverse=True)}
    pprint.pprint(genre_dict)

    plot_genre_count(genre_dict)

def analyse_keyword(df):
    # filter movies
    df = filter_movies(df)

    pprint.pprint(df)

    #create dictionary to store genre and occurence count
    keyword_dict = {}

    # use eval to convert string to list of dictionaries
    for index,row in df.iterrows():
        keywords = eval(row["keywords"])

        # loop through each genre in the list and update the dictionary
        for keyword in keywords:
            word_name = keyword["name"]
            
            if word_name in keyword_dict:
                keyword_dict[word_name] += 1
            else:
                keyword_dict[word_name] = 1    

    # get the top 40 keywords in ascending order
    keyword_dict = {k: v for k, v in sorted(keyword_dict.items(), key=lambda item: item[1], reverse=True)[:30]}

    #sort dictionary by value and print
    keyword_dict = {k: v for k, v in sorted(keyword_dict.items(), key=lambda item: item[1], reverse=True)}

    pprint.pprint(keyword_dict)
    plot_word_count(keyword_dict)

def analyse_companies(df):
    # filter movies
    df = filter_movies(df)

    pprint.pprint(df)

    #create dictionary to store genre and occurence count
    company_dict = {}

    # use eval to convert string to list of dictionaries
    for index,row in df.iterrows():
        companies = eval(row["production_companies"])

        # loop through each genre in the list and update the dictionary
        for company in companies:
            company_name = company["name"]
            
            if company_name in company_dict:
                company_dict[company_name] += 1
            else:
                company_dict[company_name] = 1    

    # get the top 40 keywords in ascending order
    company_dict = {k: v for k, v in sorted(company_dict.items(), key=lambda item: item[1], reverse=True)[:30]}

    #sort dictionary by value and print
    company_dict = {k: v for k, v in sorted(company_dict.items(), key=lambda item: item[1], reverse=True)}

    pprint.pprint(company_dict)
    plot_company_count(company_dict)

def plot_company_count(company_dict):
    #plot genre vs count as horizontal bar chart
    plt.barh(list(company_dict.keys()), company_dict.values(), color='b')
    plt.xlabel("Number of Profitable Movies")
    plt.ylabel("Production Companies")
    plt.title("Number of Profitable Movies per Production Companies")
    plt.show()

def plot_word_count(keyword_dict):
    #plot genre vs count as horizontal bar chart
    plt.barh(list(keyword_dict.keys()), keyword_dict.values(), color='b')
    plt.xlabel("Number of Popular Movies")
    plt.ylabel("Top 30 Keywords")
    plt.title("Number of Popular Movies per Keywords")
    plt.show()

def plot_genre_count(genre_dict):
    #plot genre vs count as horizontal bar chart
    plt.barh(list(genre_dict.keys()), genre_dict.values(), color='b')
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.title("Number of Movies per Genre")
    plt.show()

def analyse_linear_correlation(df, x_parameter, y_parameter):
    #plot popularity vs profit
    sns.scatterplot(x=x_parameter, y=y_parameter, data=df)

    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(x_parameter.capitalize()+" vs "+y_parameter.capitalize())

    # Train a linear regression model
    x = df[x_parameter].values.reshape(-1, 1)
    y = df[y_parameter].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    # Plot the regression line
    plt.plot(x, model.predict(x), color="red")

    #get the x value when y = 0
    print("X Intercept: ", -model.intercept_/model.coef_,"\n")

    accuracy = model.score(x, y)*100
    print("Accuracy: ", accuracy)

    plt.show()

def analyse_exp_correlation(df, x_parameter, y_parameter):
    #plot popularity vs profit
    sns.scatterplot(x=x_parameter, y=y_parameter, data=df)

    plt.xlabel(x_parameter)
    plt.ylabel(y_parameter)
    plt.title(x_parameter.capitalize()+" vs "+y_parameter.capitalize())

    # Train a linear regression model
    x = df[x_parameter].values.reshape(-1, 1)
    y = np.log(df[y_parameter].values.reshape(-1, 1))

    model = LinearRegression().fit(x, y)

    # Plot the regression line
    plt.plot(x, np.exp(model.predict(x)), color="red")

    accuracy = model.score(x, np.log(y))*100
    print("Accuracy: ", accuracy)

    plt.show()

def get_season(date):
    if date.month in (3, 4, 5):
        return 1
    elif date.month in (6, 7, 8):
        return 2
    elif date.month in (9, 10, 11):
        return 3
    else:
        return 4

def analyse_popularity_revenue(df):
    #plot popularity vs revenue
    sns.scatterplot(x="popularity", y="revenue", data=df)
    # plt.scatter(df["popularity"], df["revenue"])

    plt.xlabel("Popularity")
    plt.ylabel("Revenue")
    plt.title("Popularity vs Revenue")

    # Train a linear regression model
    x = df["popularity"].values.reshape(-1, 1)
    y = df["revenue"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    # Plot the regression line
    plt.plot(x, model.predict(x), color="red")

    plt.show()

def analyse_popularity_profit(df):
    #plot popularity vs profit
    sns.scatterplot(x="popularity", y="profit", data=df)
    # plt.scatter(df["popularity"], df["profit"])

    plt.xlabel("Popularity")
    plt.ylabel("Profit")
    plt.title("Popularity vs Profit")

    # Train a linear regression model
    x = df["popularity"].values.reshape(-1, 1)
    y = df["profit"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    # Plot the regression line
    plt.plot(x, model.predict(x), color="red")

    plt.show()

def analyse_budget_revenue(df):
    #extract average budget
    avg_budget = df["budget"].mean()
    print("Average budget: ", avg_budget)

    #plot budget vs revenue
    sns.scatterplot(x="budget", y="revenue", data=df)
    # plt.scatter(df["budget"], df["revenue"])

    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.title("Budget vs Revenue")

    # Train a linear regression model
    x = df["budget"].values.reshape(-1, 1)
    y = df["revenue"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    #for a given budget, predict the revenue
    revenue = model.predict([[avg_budget]])[0][0]
    print("Predicted revenue: ", revenue)

    # Plot the regression line
    plt.plot(x, model.predict(x), color="red")

    # plot a line for the average budget
    plt.plot([avg_budget, avg_budget], [0, revenue], '--', color="yellow",lw=2)  # dotted line from x-axis to plot
    plt.plot([0, avg_budget], [revenue, revenue], '--', color = "yellow", lw=2)

    plt.show()

def analyse_popularity_revenue_nonlinear(df, degree):
    #plot popularity vs revenue
    sns.scatterplot(x="popularity", y="revenue", data=df)
    # plt.scatter(df["popularity"], df["revenue"])

    plt.xlabel("Popularity")
    plt.ylabel("Revenue")
    plt.title("Popularity vs Revenue")

    # Reshape the arrays to fit the sklearn API
    x = df["popularity"].values.reshape(-1, 1)
    y = df["revenue"].values.reshape(-1, 1)

    # Use PolynomialFeatures to generate a matrix of polynomial features
    poly_features = PolynomialFeatures(degree)
    X_poly = poly_features.fit_transform(x)

    model = LinearRegression()
    model.fit(X_poly, y)

    # Plot the regression line using the trained model
    plt.plot(x, model.predict(X_poly), color="red")

    plt.show()

def analyse_popularity_profit_nonlinear(df, degree):
    #create new column for profit
    df["profit"] = df["revenue"] - df["budget"]

    #plot popularity vs profit
    sns.scatterplot(x="popularity", y="profit", data=df)
    # plt.scatter(df["popularity"], df["profit"])

    plt.xlabel("Popularity")
    plt.ylabel("Profit")
    plt.title("Popularity vs Profit")

    # Train a linear regression model
    x = df["popularity"].values.reshape(-1, 1)
    y = df["profit"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    # Plot the regression line
    plt.plot(x, model.predict(x), color="red")

    plt.show()

def analyse_budget_revenue_nonlinear(df, degree):
    #extract average budget
    avg_budget = df["budget"].mean()
    print("Average budget: ", avg_budget)

    #plot budget vs revenue
    sns.scatterplot(x="budget", y="revenue", data=df)
    # plt.scatter(df["budget"], df["revenue"])

    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.title("Budget vs Revenue")

    # Train a linear regression model
    x = df["budget"].values.reshape(-1, 1)
    y = df["revenue"].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    #for a given budget, predict the revenue
    revenue = model.predict([[avg_budget]])[0][0]
    print("Predicted revenue: ", revenue)

    # Plot the regression line
    plt.plot(x, model.predict(x), color="red")

    # plot a line for the average budget
    plt.plot([avg_budget, avg_budget], [0, revenue], '--', color="yellow",lw=2)  # dotted line from x-axis to plot
    plt.plot([0, avg_budget], [revenue, revenue], '--', color = "yellow", lw=2)

    plt.show()

def test_hyperparameter(df):
    #set and print axis parameters
    x_parameter = "popularity"
    y_parameter = "revenue"
    print("Testing hyperparameter for", x_parameter, "vs", y_parameter,":")

    #create new dictionary to store the accuracy scores
    scores = {}

    #split the data into training and testing sets
    train, test = train_test_split(df, test_size=0.2)

    for degree in range(1, 10):
        #print degree without newline
        print("Degree: ", degree, end=", ")
        
        # Reshape the arrays to fit the sklearn API
        x_train = train[x_parameter].values.reshape(-1, 1)
        y_train = train[y_parameter].values.reshape(-1, 1)

        # Use PolynomialFeatures to generate a matrix of polynomial features
        poly_features = PolynomialFeatures(degree)
        X_poly = poly_features.fit_transform(x_train)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_poly, y_train)

        # get the accuracy score
        x_test = test[x_parameter].values.reshape(-1, 1)
        y_test = test[y_parameter].values.reshape(-1, 1)

        X_poly_test = poly_features.fit_transform(x_test)
        
        accuracy = model.score(X_poly_test, y_test)*100
        print("Accuracy: ", accuracy)

        #store the accuracy score and the degree in the dictionary
        scores[degree] = accuracy
    
    #print the degree with the highest accuracy
    print("Best degree: ", max(scores, key=scores.get))

if __name__ == "__main__":
    main()