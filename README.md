## Title of the Research: A Customer Sentiments Analysis Tool for the Banking Sector in Kenya

## Abstract
In this study, a Design Science Research (DSR) methodology is used to develop and evaluate a sentiment analysis tool for the banking sector in Kenya. The methodology combines data preprocessing, manual annotation, topic modelling, and machine learning to fine-tune a pre-trained FinBERT model from the HuggingFace Transformers library for sentiment classification. The model's performance is evaluated using 10-fold cross-validation three times to test for the model's generalization. The results from this study show that the fine-tuned FinBERT model performs better than other baseline models in all key evaluation metrics. The results show that Latent Dirichlet Allocation (NMF) topic modelling is better than Latent Dirichlet Allocation (LDA) topic modelling in generating more coherent topics for sentiment analysis. The results from this study show that qualitative evaluation using thematic analysis and Likert scale ratings indicate a strong agreement among experts about the tool's effectiveness, reliability, and applicability, with all ratings averaging more than 4.0 on a 5-point scale. Some of the important themes that emerged include the system’s capability to provide insights, facilitate real-time decision-making, and efficiently recognize the sentiments of customers using various communication channels. In conclusion, the study shows that the integration of transformer models, aspect-based analysis, and multi-channel data sources can significantly improve the effectiveness of sentiment detection in the banking field. The study also emphasizes the need for continuous evaluation of models and the integration of user feedback in developing effective sentiment analysis systems.

## Keywords: Sentiment Analysis, Banking Sector, FinBERT, Topic Modelling, Customer Feedback, Machine Learning

## Problem Statement
Banks face challenges in efficiently identifying and responding to customer dissatisfaction due to the lack of automated, domain-specific sentiment analysis tools. Existing solutions are often limited to single data sources and fail to capture real-time, multi-channel customer interactions. This limits banks’ ability to proactively manage customer experience and improve service delivery.

## Research Questions
What is the challenge in getting information on customers’ sentiments from interactions with the banks?
What are the current tools used for sentiment analysis in banking?
What features and functionalities should a banking-specific sentiment analysis tool include?
What data sources should be integrated into the new tool?
How do end-users perceive the effectiveness and reliability of the new tool?

## Conceptual Framework (Figure 2.7)


## System Architecture Diagram (Figure 4.11)


## Description of the contents of the repository
This repository contains the implementation of a machine learning based sentiment analysis system for the Kenyan banking sector developed as part of a Master's thesis at Strathmore University.

## Link to the Streamlit App
Access the Streamlit dashboard here: https://sentiment-analysis-tool-2026.streamlit.app/ 

## Setup Instructions
Instructions on how to clone and setup the project and how to use the streamlit dashboard to validate the model

Environment Setup
1️ Clone the Repository
git clone https://github.com/KemuntoJudith/sentiment-analysis-tool.git
cd sentiment-analysis-tool

2️ Create Virtual Environment
python -m venv .venv

3️ Activate Environment
Windows (PowerShell)
.venv\Scripts\activate

Git Bash
source .venv/Scripts/activate

Mac/Linux
source .venv/bin/activate

4️ Install Dependencies
pip install -r requirements.txt

5️ Run the bankend API (Flask)


6 Run streamlit dashboard
streamlit run dashboard/dashboard.py

### Features
- Real-time sentiment analysis
- Batch feedback processing
- Aspect-based sentiment classification
- Interactive analytics dashboard
- Downloadable reports


## How to Use the Dashboard
Upload customer feedback data (CSV format)
View sentiment predictions (Positive, Neutral, Negative)
Analyze aspect-based sentiments
Monitor insights via visual dashboards
Export results for reporting
User Feedback

📋 Help improve this tool by sharing your feedback:

👉 https://forms.gle/3MofRKECigeV5bWw6 

