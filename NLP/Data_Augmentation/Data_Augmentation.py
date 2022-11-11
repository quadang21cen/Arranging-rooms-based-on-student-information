import random
import re

class StringAugmentation:
    def __init__(self) -> None:
        self.results = []
    def split_tokens(self, list_words, list_of_splits):
        # split each element in list and flatten list
        result_tokens = None
        for split_string in list_of_splits:
            if any(split_string in word for word in list_words):
                result_tokens = [(item.lower()).split(split_string) for item in list_words]
                result_tokens = [item for l in result_tokens for item in l]
        return result_tokens
    def augment(self, list_text):
        list_of_splits = [",", "và", "vừa"]
        tokens = self.split_tokens(list_text, list_of_splits)
        random.shuffle(tokens)

        return tokens

list_samples = ["abc", "abc,bde,èg", "mlk, hik, ghi"]
augment = StringAugmentation()
results = augment.augment(list_samples)
print(results)


