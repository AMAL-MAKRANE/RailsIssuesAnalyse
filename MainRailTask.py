import requests
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import os
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Set environment variable to disable oneDNN optimizations
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
    df_resampled = df.resample('ME').size()

    # Plot issue evolution over time
    plt.figure(figsize=(10, 6))
    plt.plot(df_resampled.index, df_resampled.values, marker='o')
    plt.title('Issue Evolution Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Issues')
    plt.grid(True)
    
    # Save the plot as an image
    plt.savefig('issue_evolution.png')

# Function to identify periods with more issues
def identify_periods_with_more_issues(issues):
    df = pd.DataFrame(issues)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None) 
    df['month'] = df['created_at'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    periods_with_more_issues = monthly_counts[monthly_counts > monthly_counts.mean()]
    return periods_with_more_issues

# Function to find top issue reporters
def find_top_issue_reporters(issues):
    df = pd.DataFrame(issues)
    top_reporters = df['user'].value_counts().head(5)
    return top_reporters

# Function to identify the most popular category (label)
def identify_most_popular_category(issues):
    labels = [label['name'] for issue in issues for label in issue['labels']]
    label_counts = pd.Series(labels).value_counts()
    most_popular_category = label_counts.idxmax()
    return most_popular_category

# Function to classify issues using Hugging Face's Transformers
def classify_issues(issues):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", revision="main")
    # Filter out issues with missing titles or bodies
    issues_with_text = [issue for issue in issues if issue['title'] is not None and issue['body'] is not None]
    descriptions = [issue['title'] + ' ' + issue['body'] for issue in issues_with_text]
    labels = classifier(descriptions, candidate_labels=[label['name'] for issue in issues for label in issue['labels']])
    return labels

# Function to generate PDF report
def generate_pdf_report(issues, periods_with_more_issues, top_reporters, most_popular_category):
    doc = SimpleDocTemplate("report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    report = []

   
    # Add title to the report
    report.append(Paragraph("GitHub Issues Analysis Report", styles['Title']))
    
    # Add plot to the report
    plot_img = "issue_evolution.png"
    report.append(Paragraph("Issue Evolution Over Time:", styles['Heading2']))
    report.append(Spacer(1, 220))
    report.append(Paragraph(f"<img src='{plot_img}' width='400' height='300'/>", styles['Normal']))

    # Add analysis results to the report
    report.append(Paragraph("Periods with more issues:", styles['Heading2']))
    for period, count in periods_with_more_issues.items():
        report.append(Paragraph(f"- {period}: {count} issues", styles['Normal']))

    report.append(Paragraph("Top Issue Reporters:", styles['Heading2']))
    for user, count in top_reporters.items():
        report.append(Paragraph(f"- {user}: {count} issues", styles['Normal']))

    report.append(Paragraph(f"Most Popular Category (Label): {most_popular_category}", styles['Heading2']))

    doc.build(report)

# Main function
def main():
    issues = fetch_issues()
    analyze_issue_evolution(issues)
    periods_with_more_issues = identify_periods_with_more_issues(issues)
    top_reporters = find_top_issue_reporters(issues)
    most_popular_category = identify_most_popular_category(issues)
    generate_pdf_report(issues, periods_with_more_issues, top_reporters, most_popular_category)
    classified_issues = classify_issues(issues)
    print("Issue classification:")
    print(classified_issues)
if __name__ == "__main__":
    main()

