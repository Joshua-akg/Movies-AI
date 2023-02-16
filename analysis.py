import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Read in the Movies dataset
    df = pd.read_csv("movies.csv")

    print(df.head())


if __name__ == "__main__":
    main()