import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp

# Web Scraping
@st.cache
def scrape_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    jobs_table = soup.find('table', {'id': 'jobs-table'})
    rows = jobs_table.find_all('tr')
    data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)
    columns = ['Job Title', 'Company', 'Location', 'Date Posted', 'Salary']
    jobs_df = pd.DataFrame(data, columns=columns)
    return jobs_df

# Data Cleaning and Processing
@st.cache
def clean_data(jobs_df):
    jobs_df['Date Posted'] = pd.to_datetime(jobs_df['Date Posted'])
    jobs_df['Salary'] = jobs_df['Salary'].str.replace('$', '').str.replace(',', '').astype(float)
    jobs_df = jobs_df.dropna()
    jobs_df['City'] = jobs_df['Location'].apply(lambda x: x.split(',')[0])
    jobs_df['State'] = jobs_df['Location'].apply(lambda x: x.split(',')[-1])
    return jobs_df

# Exploratory Data Analysis (EDA)
def plot_data(jobs_df):
    # Distribution of job postings by state
    st.subheader('Job Postings by State')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='State', data=jobs_df, order=jobs_df['State'].value_counts().index, ax=ax)
    st.pyplot(fig)

    # Salary distribution
    st.subheader('Salary Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(jobs_df['Salary'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

# Machine Learning Model
@st.cache
def train_model(jobs_df):
    vectorizer = CountVectorizer()
    X_title = vectorizer.fit_transform(jobs_df['Job Title'])
    X_company = vectorizer.fit_transform(jobs_df['Company'])
    X_location = vectorizer.fit_transform(jobs_df['Location'])
    X = sp.hstack([X_title, X_company, X_location])
    y = jobs_df['Salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Streamlit App
st.title('Data Science Job Postings Analysis')

url = st.text_input('Enter the URL of the job postings page', 'https://example.com/data-science-jobs')
if st.button('Scrape Data'):
    jobs_df = scrape_data(url)
    st.write('Raw Data')
    st.dataframe(jobs_df.head())
    jobs_df = clean_data(jobs_df)
    st.write('Cleaned Data')
    st.dataframe(jobs_df.head())
    plot_data(jobs_df)
    mse = train_model(jobs_df)
    st.write(f'Mean Squared Error of the Salary Prediction Model: {mse}')
st.write("Here we are at the end of getting started with streamlit! Happy Streamlit-ing! :balloon:")

