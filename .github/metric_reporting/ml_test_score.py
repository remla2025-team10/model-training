import pytest
import os

CATEGORIES = {
    "Features and Data": "Data",
    "Model Development": "Model",
    "ML Infrastructure": "Infra",
    "Monitoring": "Monitor"
}

SCORING = {
    "manual": 0.5,
    "automatic": 1
}

scores = {
    "Features and Data": 0,
    "Model Development": 0,
    "ML Infrastructure": 0,
    "Monitoring": 0
}

class MLTestScoreTestRetriever:
    def __init__(self):
        self.tests = []
    
    def pytest_collection_finish(self, session):
        for item in session.items:
            pytest_mark = item.get_closest_marker("ml_test_score")
            if pytest_mark:
                category_test = pytest_mark.kwargs.get("category_test", "")
                status = pytest_mark.kwargs.get("status", "")

                self.tests.append({
                    "name": item.name,
                    "category_test": category_test,
                    "status": status
                })

retriever = MLTestScoreTestRetriever()
pytest.main(["-q", "--collect-only", "-m", "ml_test_score", "tests/"], plugins=[retriever])

# Get all the unique category tests and their statuses
if retriever.tests:
    implemented_tests = {}

    for test in retriever.tests:
        category = test["category_test"]
        status = test["status"]
        
        if not category or not status:
            continue
        
        current_status = implemented_tests.get(category)
        if current_status is None or SCORING[status] > SCORING[current_status]:
            implemented_tests[category] = status

for category, status in implemented_tests.items():
    for category_title, category_abbreviation in CATEGORIES.items():
        if category.startswith(category_abbreviation):
            scores[category_title] += SCORING[status]
            break

FINAL_SCORE_THRESHOLD = {
    0: "More of a research project than a productionized system.",
    1: "Not totally untested, but it is worth considering the possibility of serious holes in reliability.",
    2: "There’s been first pass at basic productionization, but additional investment may be needed.",
    3: "Reasonably tested, but it’s possible that more of those tests and procedures may be automated.",
    5: "Strong levels of automated testing and monitoring, appropriate for mission-critical systems.",
    6: "Exceptional levels of automated testing and monitoring."
}

min_score = min(scores.values())

description = ""
for score, desc in FINAL_SCORE_THRESHOLD.items():
    if min_score >= score:
        description = desc
    else:
        break
    
ml_test_score_section = (
    f"## ML Test Scores\n\n"
    f"**Features and Data**: {scores['Features and Data']}\n\n"
    f"**Model Development**: {scores['Model Development']}\n\n"
    f"**ML Infrastructure**: {scores['ML Infrastructure']}\n\n"
    f"**Monitoring**: {scores['Monitoring']}\n\n"
    f"### Final ML Test Score: {min_score}\n\n"
    f"- {description}\n\n"
)

print(ml_test_score_section)

# Load the README file
with open("README.md", "r") as f:
    readme = f.read()

# If it already exists, replace the ML test score section
if "<!-- START_ML_TEST_SCORE -->" in readme and "<!-- END_ML_TEST_SCORE -->" in readme:
    before_metrics_section = readme.split("<!-- START_ML_TEST_SCORE -->")[0]
    after_metrics_section = readme.split("<!-- END_ML_TEST_SCORE -->")[1]

    new_readme = (
        before_metrics_section
        + "\n<!-- START_ML_TEST_SCORE -->\n"
        + ml_test_score_section
        + "<!-- END_ML_TEST_SCORE -->\n"
        + after_metrics_section
    )

# Otherwise add it to the bottom of README
else:
    new_readme = (
        readme
        + "\n<!-- START_ML_TEST_SCORE -->\n"
        + ml_test_score_section
        + "<!-- END_ML_TEST_SCORE -->\n"
    )

# And then update the file
with open("README.md", "w") as f:
    f.write(new_readme)