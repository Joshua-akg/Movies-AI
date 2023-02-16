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
    df = df.set_index("id")

    analyse_budget_revenue(df)

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