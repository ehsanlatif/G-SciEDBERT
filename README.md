
# G-SciEdBERT: A Contextualized LLM for Science Assessment Tasks in German

Welcome to the G-SciEdBERT repository! This project contains Jupyter notebooks for pre-training and fine-tuning the G-SciEdBERT model, an innovative large language model designed specifically for scoring German-written responses to science tasks.

## Project Overview

G-SciEdBERT is an advanced contextualized large language model (LLM) developed to automate the scoring of German science responses. Traditional models like German BERT (G-BERT) face challenges in contextual understanding, especially when dealing with complex scientific content. G-SciEdBERT addresses these limitations by incorporating domain-specific knowledge and contextual learning principles into its training process. 

### Key Features:
- **Contextual Pre-training:** G-SciEdBERT is pre-trained on a corpus of 30,000 German-written science responses from the Programme for International Student Assessment (PISA) 2018, encompassing over 3 million tokens.
- **Fine-tuning on Science Tasks:** The model is fine-tuned using 27 assessment items from PISA 2015 to enhance its scoring accuracy, achieving an 8.5% improvement in quadratic weighted Kappa over G-BERT.
- **Enhanced Scoring Accuracy:** By specializing in German science education, G-SciEdBERT significantly outperforms general-purpose models, making it a valuable tool for automated educational assessments.

## Running the Notebooks

To get started with the notebooks provided in this repository, follow the instructions below:

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Jupyter Notebook
- PyTorch
- Transformers (HuggingFace library)

### Setup Instructions

1. **Clone the Repository:**
   \`\`\`bash
   git clone https://github.com/your-repo/G-SciEdBERT.git
   cd G-SciEdBERT
   \`\`\`

2. **Install Dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Run Jupyter Notebook:**
   \`\`\`bash
   jupyter notebook
   \`\`\`

### Notebooks Description

1. **Pre_train_German_BERT.ipynb:** This notebook details the pre-training process of G-SciEdBERT using the PISA 2018 dataset. It includes data preprocessing, model training, and evaluation steps.

2. **Pre_train_German_BERT_torch.ipynb:** Similar to the first notebook but utilizes PyTorch for the pre-training process, showcasing an alternative implementation.

3. **German_BERT_MultiClass_Classification.ipynb:** This notebook demonstrates the fine-tuning process of G-SciEdBERT on the PISA 2015 dataset, detailing the steps to prepare the data, train the model, and evaluate its performance.

## Access the Model on HuggingFace

The G-SciEdBERT model is deployed on HuggingFace and can be accessed via the following link:
[G-SciEdBERT on HuggingFace](https://huggingface.co/ai4stem-uga/G-SciEdBERT)

## Citation

If you use this model or code in your research, please cite our paper:
\`\`\`
@article{Latif2023GSciEdBERT,
  title={G-SciEdBERT: A Contextualized LLM for Science Assessment Tasks in German},
  author={Latif, Ehsan and Lee, Gyeong-Geon and Neuman, Knut and Kastorff, Tamara and Zhai, Xiaoming},
  journal={arXiv preprint arXiv:2402.06584},
  year={2024}
}
\`\`\`

Thank you for using G-SciEdBERT! If you have any questions or feedback, please feel free to open an issue in this repository.
