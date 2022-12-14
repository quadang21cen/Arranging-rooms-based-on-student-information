import random
import re

class StringAugmentation:
    def __init__(self) -> None:
        self.results = []
    def split_tokens(self, list_words, list_of_splits):
        # split each element in list and flatten list
        result_tokens = list_words
        for split_string in list_of_splits:
            if any(split_string in word for word in result_tokens):
                result_tokens = [(item.lower()).split(split_string) for item in result_tokens]
                result_tokens = [item for l in result_tokens for item in l]

        return result_tokens
    def augment(self, list_text):
        list_of_splits = [",", "và", "vừa"]
        tokens = self.split_tokens(list_text, list_of_splits)
        # Clean unnecessary spaces
        tokens = [s.strip() for s in tokens]
        tokens = [re.sub(' +', ' ', s) for s in tokens]

        return tokens
if __name__ == "__main__":
    list_samples = ["abc", "xyz,sdad,sds và xyz", "sx,dds,sdds"]
    augment = StringAugmentation()
    results = augment.augment(list_samples)
    print(results)


