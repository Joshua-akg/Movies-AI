import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

def main():
    #set up seaborn 
    sns.set_theme()

    # Read in the Movies dataset
    df = pd.read_csv("movies.csv")
    # df = df.set_index("id")
    df["release_date"] = pd.to_datetime(df["release_date"])

    #get first row of df
    # print(df.iloc[0]["release_date"])

    # create new column for season of release
    df["season"] = df["release_date"].apply(get_season)
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

if __name__ == "__main__":
    main()