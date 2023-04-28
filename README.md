# Reddit Sentiment Analysis with OpenAI GPT-3

This project is a Python script that uses the Reddit API, OpenAI GPT-3, and scikit-learn to analyze the sentiment of submissions on a subreddit. The script first fetches the top 100 submissions of the past year from the specified subreddit. It then analyzes each submission's title and text with OpenAI GPT-3 to determine if the user is expressing frustration, an unexpected response, or a suggestion. If any of these sentiments is detected, the script uses OpenAI GPT-3 to generate a summary of the submission. Finally, the script uses scikit-learn to cluster similar summaries and write the final results to a JSON file.

Sample json files are included to show the output against 'r/siri'.

## Requirements

- Python 3.6+
- PRAW (Python Reddit API Wrapper)
- OpenAI API key
- scikit-learn
- tqdm

## Installation

1. Clone the repository
2. Install the requirements
3. Add your reddit and OpenAI API keys to `main.py`
4. Run the script with `python main.py <subreddit_name>`

## Usage

Run the script with `python fetch_reddit_comments.py <subreddit_name>`. Replace `<subreddit_name>` with the name of the subreddit you want to analyze.

## License

This project is licensed under the MIT License.