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
    def augment(self, list_text, num_words):
        list_of_splits = [",", "và", "vừa"]
        tokens = self.split_tokens(list_text, list_of_splits)
        random.shuffle(tokens)
        delete_gate = random.uniform(0, 1)
        
        for _ in range(num_words):
            text =', '.join(random.sample(tokens, k=random.randint(1, 4)))
            text = re.sub(' +', ' ', text)  # Remove more than one space
            self.results.append(text)
        return self.results

list_samples = ["Tôi đã làm tốt và học tập", "Tôi làm chức tổng giám đốc", "Quà, Na, Ngọc và Hạ"]
augment = StringAugmentation()
results = augment.augment(list_samples, num_words = 10)
print(results)


