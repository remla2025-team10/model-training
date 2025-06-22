import json

# First we load the metrics
with open("metrics/metrics.json", "r") as f:
    performance_metrics = json.load(f)


# Then extract them (and round because they have many decimals)
accuracy = round(performance_metrics["accuracy"], 2)
precision = round(performance_metrics["precision"], 2)
recall = round(performance_metrics["recall"], 2)
f1_score = round(performance_metrics["f1_score"], 2)
confusion_matrix = performance_metrics["confusion_matrix"]

performance_metrics_section = (
    f"## Performance Metrics\n\n"
    f"- **Accuracy**: {accuracy}\n"
    f"- **Precision**: {precision}\n"
    f"- **Recall**: {recall}\n"
    f"- **F1 Score**: {f1_score}\n\n"
    f"- **Confusion Matrix**: {confusion_matrix}\n\n"
)

# Load the README file
with open("README.md", "r") as f:
    readme = f.read()


# If it already exists, replace the performance metrics section
if "<!-- START_PERFORMANCE_METRICS -->" in readme and "<!-- END_PERFORMANCE_METRICS -->" in readme:
    before_metrics_section = readme.split("<!-- START_PERFORMANCE_METRICS -->")[0]
    after_metrics_section = readme.split("<!-- END_PERFORMANCE_METRICS -->")[1]

    new_readme = (
        before_metrics_section
        + "\n<!-- START_PERFORMANCE_METRICS -->\n"
        + performance_metrics_section
        + "<!-- END_PERFORMANCE_METRICS -->\n"
        + after_metrics_section
    )

# Otherwise add it to the bottom of README
else:
    new_readme = (
        readme
        + "\n<!-- START_PERFORMANCE_METRICS -->\n"
        + performance_metrics_section
        + "<!-- END_PERFORMANCE_METRICS -->\n"
    )

# And then update the file
with open("README.md", "w") as f:
    f.write(new_readme)