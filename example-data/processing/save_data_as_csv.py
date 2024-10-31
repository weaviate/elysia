import json
import pandas as pd

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def remove_comments(text):
    lines = text.split("\n")
    for line in lines:
        if line.startswith("<!--"):
            lines.remove(line)
    return "\n".join(lines)

def remove_empty_headers(text):
    lines = text.split("\n")
    newlines = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("##"):
            j = i + 1
            while j < len(lines):
                if len(lines[j].strip()) == 0:
                    j += 1
                elif lines[j].startswith("##"):
                    i = j - 1
                    break
                else:
                    newlines.append(lines[i])
                    break
            else:
                newlines.extend(lines[i:j])

        else:
            newlines.append(lines[i])
        i += 1
    return "\n".join(newlines)

def remove_empty_lines(text):
    lines = text.split("\n")
    return "\n".join([line for line in lines if len(line.strip()) > 0])

if __name__ == "__main__":
    with open("verba_github_issues.json", "r") as f:
        data = json.load(f)

    for i in range(len(data)):
        if data[i]["body"] is not None:
            text = remove_comments(data[i]["body"])
            text = remove_empty_headers(text)
            text = remove_empty_lines(text)
            data[i]["body"] = text

    df = pd.DataFrame(data)
    df.to_csv("verba_github_issues.csv", index=False)
