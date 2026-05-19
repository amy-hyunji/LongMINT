import re
import string
from num2words import num2words


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def convert_numbers_to_words(text: str) -> str:
        """Convert standalone digit numbers in text to words."""
        def repl(match):
            num = int(match.group(0))
            return num2words(num)
        return re.sub(r"\b\d+\b", repl, text)


    return white_space_fix(remove_articles(remove_punc(lower(convert_numbers_to_words(str(answer))))))
