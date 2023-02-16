import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

def main():
    # Read in the Movies dataset
    df = pd.read_csv("movies.csv")
    df = df.set_index("id")

    #print column heads
    print(df.head(),"\n")

    #plot budget vs revenue
    plt.scatter(df["budget"], df["revenue"])
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.title("Budget vs Revenue")

    plt.show()

if __name__ == "__main__":
    main()