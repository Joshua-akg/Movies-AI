import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures    

def main():
    #set up seaborn 
    sns.set_theme()

    # Read in the Movies dataset
    df = pd.read_csv("movies.csv")

    #create new column for profit
    df["profit"] = df["revenue"] - df["budget"]

    # analyse_popularity_profit_nonlinear(df, 6)

    test_hyperparameter(df)

    # df = df.set_index("id")
    # df["release_date"] = pd.to_datetime(df["release_date"])

    #get first row of df
    # print(df.iloc[0]["release_date"])

    # create new column for season of release
    # df["season"] = df["release_date"].apply(get_season)
    # print(df["season"])

    # model = LinearRegression()
    # model.fit(df[["season"]], df["revenue"])

    # X = df[['season']]
    # y = df['revenue']
    # y_pred = model.predict(X)

    # plt.scatter(X, y, color='blue')
    # plt.plot(X, y_pred, color='red', linewidth=2)

    # plt.xlabel('Season of release')
    # plt.ylabel('Revenue')
    # plt.show()

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
    print("Testing hyperparameter for ", x_parameter, " vs ", y_parameter,":")

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