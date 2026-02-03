# Heart Disease Classification using Neural Networks

This project implements a **Multi-Layer Perceptron (MLP) Neural Network** to predict the presence of heart disease using the Cleveland and Hungarian datasets from the UCI Machine Learning Repository.

##  Key Features
*   *Merges multiple datasets (Cleveland & Hungarian) for a robust training set.
*   *Handles missing values using linear interpolation and cleans data for optimal model performance.
*   *Feature Scaling*: Utilizes `StandardScaler` to normalize input features, ensuring stable and fast convergence for the Neural Network.
*   **Model Architecture**: A deep learning model with:
    *   **300 Hidden Neurons**: For capturing complex non-linear patterns.
    *   **ReLU Activation**: To prevent the vanishing gradient problem.
    *   **Adam Solver**: For efficient stochastic gradient-based optimization.
*   **Custom Evaluation**: Includes a custom implementation of Confusion Matrix metrics (TP, FP, TN, FN) and visualizes them using Matplotlib.

##  Technologies Used
*   Python 3.x
*   Pandas
*   NumPy
*   Scikit-Learn
*   Matplotlib

##  Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dhickey/heart-disease-ai.git
    cd heart-disease-ai
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

3.  **Data Setup:**
    *   This project uses the **Processed Cleveland** and **Hungarian** data files.
    *   Ensure `processed.cleveland.data` and `processed.hungarian.data` are available locally.
    *   Note: You need to update the file paths in `AI project.py` (lines 27-28) to match your local directory structure.

##  Usage
Run the main script to train the model and see the evaluation results:

```bash
python "AI project.py"
```

The script will:
1.  Load and clean the data.
2.  Train the Neural Network.
3.  Output the training accuracy.
4.  Display a bar chart visualizing True Positives, False Positives, True Negatives, and False Negatives.
5. Evaluate performance on a test set and provide detailed metrics on its predictive capability for heart disease diagnosis.
