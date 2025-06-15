import re
import sys

score = None

with open("pylint.log", "r") as f:
    content = f.read()

match = re.search(r"Your code has been rated at (-?\d+\.\d+)\/10", content)
if match:
    score = float(match.group(1))
else:
    print("No score found in pylint.log")

if score is not None:
    color = "brightgreen" if score >= 8 else "yellow" if score >= 6 else "red"
    score_value = f"{score:.2f}"
    badge_url = f"https://img.shields.io/badge/Pylint%20Score-{score_value}%2F10-{color}"
    badge_content = f"![Pylint Score]({badge_url})\n"

    with open("README.md", "r") as f:
        readme = f.read()

    if "<!-- START_PYLINT_SCORE -->" in readme and "<!-- END_PYLINT_SCORE -->" in readme:
        before_section = readme.split("<!-- START_PYLINT_SCORE -->")[0]
        after_section = readme.split("<!-- END_PYLINT_SCORE -->")[1]

        new_readme = (
            before_section
            + "\n<!-- START_PYLINT_SCORE -->\n"
            + badge_content
            + "<!-- END_PYLINT_SCORE -->\n"
            + after_section
        )
    else:
        new_readme = (
            readme
            + "\n<!-- START_PYLINT_SCORE -->\n"
            + badge_content
            + "<!-- END_PYLINT_SCORE -->\n"
        )

    with open("README.md", "w") as f:
        f.write(new_readme)
    print(f"Updated README.md with Pylint score: {score_value}")

else:
    print("No Pylint score found to update README.md")