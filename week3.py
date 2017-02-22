def bag_of_words(text):
    # TODO: Implement bag of words
    d = {}
    for wrd in text.split():
        if wrd not in d.keys():
            d[wrd] = 1
        else:
            d[wrd] += 1
    return d

test_text = 'the quick brown fox jumps over the lazy dog'
print(bag_of_words(test_text))

from collections import Counter

def bag_word(text):
    return Counter(text.split())

print(bag_word(test_text))
