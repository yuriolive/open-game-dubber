import json
import os
import subprocess
import urllib.parse
import urllib.request


def get_pr_diff():
    # Fetch base branch to compare against
    subprocess.run(["git", "fetch", "origin", os.environ["GITHUB_BASE_REF"]], check=True)
    # Get the diff
    result = subprocess.run(
        ["git", "diff", f"origin/{os.environ['GITHUB_BASE_REF']}...HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout


def generate_description(diff, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    prompt = f"""You are an expert software engineer.
Review the following git diff and generate a concise, professional Pull Request description.
Format the output in Markdown with the following sections:
- **Summary**: A brief overview of the changes.
- **Key Changes**: A bulleted list of the most important modifications.
- **Impact**: How these changes affect the project.

Git Diff:
{diff[:15000]} # Truncate if too large
"""

    data = {"contents": [{"parts": [{"text": prompt}]}]}

    req = urllib.request.Request(
        url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )

    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["candidates"][0]["content"]["parts"][0]["text"]


def update_pr_description(description, github_token, repo, pr_number):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"

    data = {"body": description}

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="PATCH",
    )

    with urllib.request.urlopen(req) as response:
        if response.status == 200:
            print("Successfully updated PR description.")
        else:
            print(f"Failed to update PR description: {response.status}")


def main():
    try:
        if not os.environ.get("GITHUB_EVENT_PATH"):
            print("Not running in GitHub Actions.")
            return

        with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
            event = json.load(f)

        pr_number = event["pull_request"]["number"]
        repo = os.environ["GITHUB_REPOSITORY"]
        github_token = os.environ["GITHUB_TOKEN"]
        api_key = os.environ["GEMINI_API_KEY"]

        print(f"Generating description for PR #{pr_number}...")
        diff = get_pr_diff()

        if not diff.strip():
            print("No changes found.")
            return

        description = generate_description(diff, api_key)
        update_pr_description(description, github_token, repo, pr_number)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
