# QA Set Custom Trainer

This project implements a custom trainer for the Hugging Face library, featuring a unique loss function designed for a specific task. The custom loss function calculates the loss for a set of answers associated with a single sentence and predicate, regardless of the order of the answers.

## Installation

1. **Dataset**:
   - The dataset used to train this model can be found at the following link: [QA-SRL Dataset](https://nlp.biu.ac.il/~ron.eliav/qasrl/V-passive_red/).
   - An example demonstrating how to download, format, and use the dataset is provided in the `ElinoyMSprojectExample.ipynb` notebook.

2. **Dependencies**:
   - To install the required packages, run the following command:
     ```bash
     pip install -r requirements.txt
     ```

3. **Environment**:
    - The code was developed in Google Colab, which provides a free GPU environment, for most of the research.
    - The code also was tested on the university's linux server, which provides a GPU environment.
      - In the server, I ran the code using anaconda installation with jupyter notebook.
      - helpful guides for convenient run on the server:
        - [Anaconda installation](https://www.anaconda.com/download/success)
        - [Running IPython/Jupyter notebook on different computer](https://danielhnyk.cz/running-ipython-notebook-different-computer/)
        - [how-to-use-linux-screen](https://linuxize.com/post/how-to-use-linux-screen/)
        

## Usage

1. **Workflow Example**:
   - A full example of the usage flow is provided in the `ElinoyMSprojectExample.ipynb` notebook.
   - The notebook includes steps to train the model on the QA-SRL dataset and compare the results to a baseline model using the cross-entropy (CE) loss.

2. **Steps for Using the Custom Trainer**:
   - **Data Loading and Preprocessing**:
     - Use the `QASrl.py` script to load the QA-SRL dataset.
     - Preprocess the data using the `format_input` function in the notebook.
   - **Set Global Default Values**:
     - Run the `set_global_default_values` function to configure global variables before training.
       I've set those as global variables to reduce the number of calls to function and of calculations, in order to improve performance, and because I need to use their values in compute_matrics and in the custom loss function and we can't pass additional arguments to those functions.
   - **Trainer Configuration**:
     - Pass the following additional parameters to the trainer constructor (default values are defined in `defaultValues.py`):
       - `lambda1`: Weight of the CE loss in the total loss.
       - `lambda2`: Weight of the custom loss in the total loss.
       - `qa_sep`: Tensor representing the tokens that separate different QAs in the input (already on CUDA if available).
       - `q_sep`: Tensor representing the tokens that separate the question from the answers in the input (already on CUDA if available).
       - `a_sep`: Tensor representing the tokens that separate different answers of a question (already on CUDA if available).
       - `padding_idx`: The index of the padding token in the tokenizer.
       - `use_device`: The device to run the model on (e.g., CUDA or CPU).
   - **Training**:
     - Train the model using the custom trainer.
   - **Evaluation**:
     - Evaluate the model’s performance using the custom evaluation metrics implemented in `computeQASetValidationMetrics.py`.

## Evaluation Metrics

This project also introduces custom evaluation metrics tailored for the QA-SRL answers set task. These metrics assess the model’s performance on a set of answers associated with a single sentence and predicate. The metrics include:

- **Accuracy**: The ratio of correct answers to the total number of answers.
- **Precision**: The ratio of correct answers to the total number of predicted answers.
- **Recall**: The ratio of correct answers to the total number of true answers.
- **F1-Score**: The harmonic mean of precision and recall.

The evaluation metrics are implemented in the `compute_metrics` function within `computeQASetValidationMetrics.py`. This function:

- Receives the model’s predictions and the true answers.
- Splits and aligns the answers.
- Calculates metrics based on an Intersection over Union (IoU) threshold of 0.5, counting answers with IoU > 0.5 as correct.
