# Import necessary libraries
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import math


def read_shapefile(shapefile_path):
    """
    Reads a shapefile and returns a GeoDataFrame.

    Parameters:
    - shapefile_path (str): The file path to the shapefile.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the spatial data from the shapefile.
    """
    shapefile_path = "../data/cb_2018_us_county_5m.shp"
    return gpd.read_file(shapefile_path)


def preprocess_data(county_aggregation):
    """
    Preprocesses the county aggregation data, preparing it for merging with spatial data.

    Parameters:
    - county_aggregation (pd.DataFrame): The DataFrame containing county aggregation data.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame with necessary modifications for further analysis.
    """
    county_aggregation = pd.read_csv("../data/countyagg.csv")
    # Convert CensusTract to string and generate FIPS code
    county_aggregation["CensusTract"] = county_aggregation["CensusTract"].astype(str)
    county_aggregation["FIPS"] = county_aggregation["CensusTract"].str[:5]
    return county_aggregation


def merge_data(gdf, county_aggregation):
    """
    Merges spatial data in a GeoDataFrame with county aggregation data on a common key.

    Parameters:
    - gdf (gpd.GeoDataFrame): The GeoDataFrame containing spatial data.
    - county_aggregation (pd.DataFrame): The DataFrame containing county aggregation data.

    Returns:
    - gpd.GeoDataFrame: A merged GeoDataFrame containing both spatial and aggregation data.
    """
    merged_gdf = gdf.merge(
        county_aggregation, left_on="GEOID", right_on="FIPS", how="left"
    )
    return merged_gdf


def plot_states(merged_gdf, feature_to_plot):
    """
    Plots the specified feature for each state in a grid layout using a GeoDataFrame.

    Parameters:
    - merged_gdf (gpd.GeoDataFrame): The merged GeoDataFrame containing geospatial and feature data.
    - feature_to_plot (str): The name of the feature column to be plotted.

    User Input:
    - feature_to_plot (str): The name of the feature to plot from a list of available features.

    Displays:
    - A series of plots for each state showing the specified feature.
    """
    states = merged_gdf["State"].unique()
    n = len(states)
    cols = 3  # Number of columns in the grid layout
    rows = math.ceil(n / cols)  # Number of rows calculated based on number of states

    fig, axes = plt.subplots(
        rows, cols, figsize=(20, rows * 6), constrained_layout=True
    )
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, state in enumerate(states):
        state_gdf = merged_gdf[merged_gdf["State"] == state]
        if state_gdf.empty:
            continue

        # Use the feature_to_plot for coloring the plot
        state_gdf.plot(
            column=feature_to_plot,
            ax=axes[i],
            legend=True,
            legend_kwds={
                "label": f"{feature_to_plot} by County",
                "orientation": "horizontal",
            },
        )
        axes[i].set_title(f"{state} - {feature_to_plot}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")  # Turn off any unused subplots

    plt.show()


def main():
    shapefile_path = "../data/cb_2018_us_county_5m.shp"
    gdf = read_shapefile(shapefile_path)

    # Load your county_aggregation data here
    county_aggregation = pd.read_csv("../data/countyagg.csv")

    county_aggregation = preprocess_data(county_aggregation)
    merged_gdf = merge_data(gdf, county_aggregation)

    # List available features for plotting
    print("Available features to plot:")
    for column in merged_gdf.columns:
        print(column)

    # Prompt the user to select a feature
    feature_to_plot = input("Enter the name of the feature to plot: ").strip()

    # Validate the input (optional)
    if feature_to_plot not in merged_gdf.columns:
        print(f"'{feature_to_plot}' is not a valid feature name. Try again.")
        return

    plot_states(merged_gdf, feature_to_plot)


if __name__ == "__main__":
    main()
