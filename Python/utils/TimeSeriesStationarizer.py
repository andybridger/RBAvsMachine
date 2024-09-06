import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import polars as pl
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class TimeSeriesStationarizer:
    def __init__(self, df, series=None, series_id=None, ):
        """
        Initialize the TimeSeriesStationarizer with a DataFrame and the name of the series to stationarize.
        Assumes the DataFrame has a 'series_id' column to filter by the series name and a 'value' column for the data.

        :param df: pandas DataFrame with columns ['series_id', 'date', 'value'].
        :param series_name: The value in 'series_id' column to filter the DataFrame by.
        """
        self.series_id = series_id
        self.series_name = series
        # Filter the DataFrame for the selected series
        if self.series_name is not None:
            self.df_selected = df[df['series'] == series][['date', 'value']].copy().reset_index(drop=True)
        elif self.series_id is not None:
            self.df_selected = df[df['series_id'] == self.series_id][['date', 'value']].copy().reset_index(drop=True)
        else:
            raise SyntaxError("Please pass an argument to `series` or `series_id`.")

        self.transformations = [
            ("original", lambda x: x),
            ("Dx", lambda x: x.diff()),
            ("D2x", lambda x: x.diff().diff()),
            ("Lx", lambda x: np.log(x)),
            ("DLx", lambda x: np.log(x).diff()),
            ("D2Lx", lambda x: np.log(x).diff().diff()),
            ("Dxx", lambda x: (x / x.shift() - 1).diff())
        ]

    def transform(self, transformation_name):
        """
        Apply the transformation to the 'value' column of the filtered DataFrame.

        :param transform: A lambda string that describes to the time series.
        :return: pandas DataFrame with an additional 'transformed' column.
        """
        # Retrive Transformation function
        t_function = self.find_transformation(transformation_name)
        transformed_df = self.apply_transformation(t_function)
        return self.df_selected

    def apply_transformation(self, transform_func):
        """
        Apply the transformation function to the 'value' column of the filtered DataFrame.

        :param transform_func: A lambda function that applies a transformation to a pandas Series.
        :return: pandas DataFrame with an additional 'transformed' column.
        """
        self.df_selected['transformed'] = transform_func(self.df_selected['value'])
        return self.df_selected

    def test_stationarity(self, series):
        """
        Test if a given pandas Series is stationary using the Augmented Dickey-Fuller test.

        :param series: pandas Series to test.
        :return: Boolean indicating if the series is stationary (True) or not (False).
        """
        series_dropna = series.dropna()  # Dropping NA values before the test
        result = adfuller(series_dropna, autolag='AIC')
        p_value = result[1]
        return p_value <= 0.05  # Common threshold for stationarity

    def stationarize(self):
        """
        Apply transformations to stationarize the series and return the transformation that achieves stationarity.

        :return: Tuple of (pandas DataFrame with the stationarized series, name of the transformation).
        """
        for name, transform in self.transformations:
            transformed_df = self.apply_transformation(transform)
            if self.test_stationarity(transformed_df['transformed']):
                print(f"Transformation '{name}' achieved stationarity.")
                return transformed_df, name
        print("No transformation achieved stationarity.")
        return self.df_selected[['date', 'value']], "original"

    def stationarize_all(self):
        """
        Apply all transformations to the series, test each for stationarity, and print a summary table.

        :return: A pandas DataFrame with the summary of stationarity tests for each transformation.
        """
        results = []  # To store the results of stationarity tests

        for name, transform in self.transformations:
            transformed_df = self.apply_transformation(transform)
            is_stationary = self.test_stationarity(transformed_df['transformed'])
            results.append({
                'Transformation': name,
                'Stationary': 'Yes' if is_stationary else 'No'
            })

        # Convert the results list to a DataFrame for display
        results_df = pd.DataFrame(results)

        # Print the summary table
        print("Summary of Stationarity Tests:")
        print(results_df)

        self.plot_series()

        return results_df

    def plot_series(self, transformation='original', plot_orig=True):
        """
        Plot the original series and the transformed series based on the provided transformation name.

        :param transformation: Name of the transformation to plot alongside the original series.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        transformed_df = self.transform(transformation)

        # Plot the original series
        if plot_orig:
            ax.plot(transformed_df['date'], transformed_df['value'], label='Original Series', marker='o')

        if transformation != 'original':
            # Ensure the transformation exists in the DataFrame
            if 'transformed' in self.df_selected.columns:
                ax.plot(transformed_df['date'], transformed_df['transformed'],
                        label=f'Transformed Series ({transformation})', marker='x')
            else:
                print(f"No transformed series found for '{transformation}'. Showing only the original series.")
        else:
            print("Showing only the original series.")

        ax.set_title(f"Time Series: Original vs. {transformation}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def find_transformation(self, transformation_name):
        """
        Method that returns the transformation function from the list of transformations
        """
        for name, func in self.transformations:
            if name == transformation_name:
                return func
        return None


