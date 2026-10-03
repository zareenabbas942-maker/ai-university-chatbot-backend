import re


def split_university_data(text):

    # Split the university knowledge base
    # whenever a new uppercase section heading appears.

    pattern = r"\n(?=[A-Z][A-Z &/\-()0-9]+(?:\n|:))"

    sections = re.split(pattern, text)

    chunks = []

    for section in sections:

        section = section.strip()

        if section:

            chunks.append(section)

    return chunks