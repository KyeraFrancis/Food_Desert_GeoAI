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
- **Time-Series Analysis**: Utilize the temporal aspect of the dataset to showcase trends over time, primarily focused at the state level. (`Future Work`)

## Dataset
The base dataset for this project is aggregated at the state level, sourced from the United States Department of Agriculture (USDA). This dataset provides a solid foundation for initial analysis and model training.
Due to size of Dataset, it is not included in the repository.
[Link to Dataset](https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/)

## Repository Structure
- `data/`: Contains the datasets used in the project.
- `src/`: Source code for the project.
  - `logreg.py`: Logistic regression model.
  - `geopd.py`: Geospatial data analysis and mapping.
  - `RFC.py`: Random Forest Classifier model.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and initial testing.
  - `EDA`: Main Jupyter Notebook with EDA, Visualization and Modeling.
  - `Modeling`: Jupyter Notebook with Modeling from the src folder.
  - `Visualization`: Jupyter Notebook with Data Analysis/Visualization.
- `docs/`: Additional documentation, reports, Data Dictionary.
- `README.md`: Overview and instructions for the project.
- `requirements.txt`: List of dependencies for the project.

## Installation and Usage
This project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- geopandas
- scikit-learn

To install these dependencies, you can use the `requirements.txt` file included in this repository. First, ensure you have Python and pip installed on your system. Then, run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Contributing
We welcome contributions from the community, whether it's adding new models, enhancing the analysis with data from previous years, or improving the existing codebase. Here's how you can contribute:

### Getting Started

1. **Fork the Repository**: Begin by forking the repository to your GitHub account. This creates a copy of the repository where you can make your changes.

2. **Clone the Forked Repository**: Clone the repository to your local machine to work on the project. You can do this by running:
   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY 
   ```
    Replace `YOUR-USERNAME` with your GitHub username and `YOUR-REPOSITORY` with the name of the repository.
  Be sure to replace `YOUR-USERNAME` and `YOUR-FORKED-REPO` with your GitHub username and the name of your forked repository, respectively.

3. **Create a New Branch**: It's best to make your changes in a new branch. Navigate to the project directory on your machine and create a branch using:
    ```bash
    git checkout -b branch-name
    ```
    Replace `branch-name` with a descriptive name for your branch.
Replace `your-branch-name` with a name relevant to the changes you're making.

### Making Changes

- **Add or Update Models**: If you're contributing by adding a new model or updating an existing one, please ensure your code is clear and well-documented.

- **Incorporate Data from Previous Years**: The dataset required for analysis needs to be downloaded locally. If you're adding data from previous years, please download it from the link provided at the beginning of the README. Make sure to follow any existing data naming conventions and update file paths in the code as necessary.

- **Test Your Changes**: Before submitting your changes, thoroughly test your code to ensure it functions as expected.

### Submitting Your Contributions

1. **Commit Your Changes**: Once you're ready to submit your changes, commit them to your branch with a descriptive message:

    ```bash
    git add .
    git commit -m "Your commit message"
    ```

2. **Push to GitHub**: Push your changes to your fork on GitHub:
  
      ```bash
      git push origin branch-name
      ```

3. **Create a Pull Request**: Go to the original project repository on GitHub. You should see an option to "Compare & pull request" from your branch. Click on it, fill in the details of your changes, and submit the pull request.

### Review Process

Once submitted, your pull request will be reviewed. You may receive feedback or requests for changes to your contribution. This is a standard part of the collaboration process, so don't be discouraged!

By following these guidelines, you can help enhance this project and contribute to its growth. We look forward to your contributions and are excited to see how the project evolves with the community's input.

If you have any questions about contributing, please feel free to reach out by opening an issue in the repository.

These are common guidelines for contributing to open source projects. If you're new to open source, you can also refer to GitHub's guide on [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/).

## License
MIT License

Copyright (c) 2024 Kyera Francis

## Acknowledgments
General Assembly Instructors and TAs:
- Hank Butler, Alanna Besaw
- Bryan Ortiz
