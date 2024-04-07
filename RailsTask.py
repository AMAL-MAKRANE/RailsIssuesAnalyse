import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import os
import csv
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to fetch issues from GitHub repository
def fetch_issues():
    url = 'https://api.github.com/repos/rails/rails/issues'
    params = {'per_page': 100, 'page': 1}  # Fetch 100 issues per page
    issues = []

    for _ in range(5):  # Fetch 5 pages to get 500 issues
        response = requests.get(url, params=params)
        data = response.json()
        issues.extend(data)
        if 'next' in response.links:
            url = response.links['next']['url']
        else:
            break

    return issues

# Function to analyze issue evolution over time
def analyze_issue_evolution(issues):
    df = pd.DataFrame(issues)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    df.resample('M').size().plot(title='Issue Evolution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.show()

# Function to identify periods with more issues
def identify_periods_with_more_issues(issues):
    df = pd.DataFrame(issues)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['month'] = df['created_at'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    periods_with_more_issues = monthly_counts[monthly_counts > monthly_counts.mean()]
    print("Periods with more issues:")
    print(periods_with_more_issues)

# Function to find top issue reporters
def find_top_issue_reporters(issues):
    df = pd.DataFrame(issues)
    top_reporters = df['user'].value_counts().head(5)
    print("Top Issue Reporters:")
    print(top_reporters)

# Function to identify the most popular category (label)
def identify_most_popular_category(issues):
    labels = [label['name'] for issue in issues for label in issue['labels']]
    label_counts = pd.Series(labels).value_counts()
    most_popular_category = label_counts.idxmax()
    print("Most Popular Category (Label):", most_popular_category)

# Function to classify issues using Hugging Face's Transformers
def classify_issues(issues):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", revision="main")
    # Filter out issues with missing titles or bodies
    issues_with_text = [issue for issue in issues if issue['title'] is not None and issue['body'] is not None]
    descriptions = [issue['title'] + ' ' + issue['body'] for issue in issues_with_text]
    labels = classifier(descriptions, candidate_labels=[label['name'] for issue in issues for label in issue['labels']])
    return labels


# Main function
def main():

    issues = fetch_issues()
    analyze_issue_evolution(issues)
    identify_periods_with_more_issues(issues)
    find_top_issue_reporters(issues)
    identify_most_popular_category(issues)
    # Write issues data to a CSV file
    with open('issues.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'title', 'body', 'created_at', 'updated_at', 'labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for issue in issues:
            filtered_issue = {key: issue[key] for key in fieldnames}
            writer.writerow(filtered_issue)

    # Classify issues
    classified_issues = classify_issues(issues)
    print("Issue classification:")
    print(classified_issues)

if __name__ == "__main__":
    main()
