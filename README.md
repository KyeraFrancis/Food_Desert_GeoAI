# Food Desert GeoAI Analysis

## Overview
The Food Desert GeoAI Project is an innovative approach to mapping and analyzing food deserts, areas with limited access to affordable and nutritious food. This project aims to identify the demographics most affected by food deserts and analyze the impacts on local communities. By leveraging geospatial data analysis and machine learning models, this project seeks to uncover patterns and trends in the distribution and characteristics of food deserts. This can be used to inform policy decisions and community initiatives to address food insecurity and improve access to healthy food.

## Objectives
- **Mapping Food Deserts**: Develop an application to visually represent the geographic distribution of food deserts.
- **Demographic Analysis**: Identify and analyze the demographics disproportionately affected by food deserts.
- **Impact Assessment**: Evaluate the effects of food deserts on local communities.
- **Model Development**:
  - Utilize a Logistic Regression model to classify areas as food deserts or non-food deserts.
  - Employ Clustering models to discover patterns within food deserts.
  - Conduct Geospatial Data Analysis to analyze spatial data and visualize trends.
- **Time-Series Analysis**: Utilize the temporal aspect of the dataset to showcase trends over time, primarily focused at the state level.

## Dataset
The base dataset for this project is aggregated at the state level, sourced from the United States Department of Agriculture (USDA). This dataset provides a solid foundation for initial analysis and model training. As the project progresses, additional data from states with the highest and lowest prevalence of food deserts may be included for comparative analysis.

[Link to Dataset](https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/)

## Repository Structure
- `data/`: Contains the datasets used in the project.
- `src/`: Source code for the project.
  - `logistic_regression/`: Logistic regression model.
  - `geospatial_analysis/`: Geospatial data analysis and mapping.
  - `clustering/`: Clustering model (stretch goal).
- `notebooks/`: Jupyter notebooks for exploratory data analysis and initial testing.
- `docs/`: Additional documentation, reports, or relevant research papers.
- `README.md`: Overview and instructions for the project.
- `requirements.txt`: List of dependencies for the project.

## Installation and Usage
(Place Holder:

Instructions on how to set up and use the project, including installing dependencies, setting up a virtual environment, and running the application.)

## Contributing
(Place Holder:

Guidelines for how others can contribute to the project. This might include instructions for forking the repository, creating a feature branch, making changes, and submitting a pull request.)

## License
MIT License

Copyright (c) 2024 Kyera Francis

## Acknowledgments
General Assembly Instructors and TAs:
- Hank Butler, Alanna Besaw
- Bryan Ortiz
