import praw
import sys
import json
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

REDDIT_CLIENT_ID = "<your_client_id>"
REDDIT_CLIENT_SECRET = "<your_client_secret>"
REDDIT_USER_AGENT = "<your_user_agent>"
OPENAI_API_KEY = "<your_openai_api_key>"

def gpt3_analyze_comment(api_key, text):
    openai.api_key = api_key
    prompt = (
        f"Please determine if the following comment contains either a suggestion that a company should fix something,"
        f" if something happened that did not meet the user expectations,"
        f" or if the user is expressing frustration:\n\n```{text}```\n\n"
        f"Suggestion: {{'True' if suggestion else 'False'}}\n"
        f"Unexpected: {{'True' if suggestion else 'False'}}\n"
        f"Frustrated: {{'True' if frustration else 'False'}}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    analysis_output = response.choices[0].message['content'].strip()
    is_suggestion = "True" in analysis_output.split("Suggestion:")[-1].split("\n")[0]
    is_unexpected = "True" in analysis_output.split("Unexpected:")[-1].split("\n")[0]
    is_frustrated = "True" in analysis_output.split("Frustrated:")[-1].split("\n")[0]

    return {"is_suggestion": is_suggestion,  "is_unexpected": is_unexpected, "is_frustrated": is_frustrated}

def gpt3_summarize_submission(api_key, text):
    openai.api_key = api_key
    prompt = (f"Please summarize the following comment in one sentence," 
              f"either starting with \"Suggestion:\" or \"Unexpected:\" or \"Frustrated:\" as appropriate:\n\n{text}\n\nSummary:"
              )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].message['content'].strip()

def cluster_summaries(summaries):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(summaries)
    true_k = 10 # set number of clusters
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    labels = model.predict(X)
    clusters = {}
    for i in range(len(summaries)):
        label = labels[i]
        if label not in clusters:
            clusters[label] = [i]
        else:
            clusters[label].append(i)
    return clusters


def main():
    if len(sys.argv) != 2:
        print("Usage: python fetch_reddit_comments.py <subreddit_name>")
        sys.exit(1)

    subreddit_name = sys.argv[1]

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    subreddit = reddit.subreddit(subreddit_name)
    top_submissions = list(subreddit.top(time_filter="year", limit=100))

    submissions_summary = []
    for i, submission in enumerate(top_submissions):
        print(f"Processing submission {i+1} of {len(top_submissions)}")
        analysis = gpt3_analyze_comment(OPENAI_API_KEY, submission.title + ' ' + submission.selftext)
        is_suggestion = analysis["is_suggestion"]
        is_unexpected = analysis["is_unexpected"]
        is_frustrated = analysis["is_frustrated"]

        if is_suggestion or is_unexpected or is_frustrated:
            summary = gpt3_summarize_submission(OPENAI_API_KEY, submission.title + ' ' + submission.selftext)
            submissions_summary.append({"submission_id": submission.id, "title": submission.title, "summary": summary})

    with open("submissions_summary.json", "w") as f:
        json.dump(submissions_summary, f, indent=2)

    clusters = cluster_summaries([submission["summary"] for submission in submissions_summary])
    deduplicated_summaries = []
    for cluster in clusters.values():
        deduplicated_summaries.append(submissions_summary[cluster[0]])

    with open("submissions_dedup.json", "w") as f:
        json.dump(deduplicated_summaries, f, indent=2)

    print("Deduplicated summaries saved to submissions_dedup.json")

if __name__ == "__main__":
    main()
