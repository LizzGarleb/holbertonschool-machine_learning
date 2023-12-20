#!/usr/bin/env python3

tf_idf = __import__('1-tf_idf').tf_idf

sentences = ["Holberton school is Awesome!",
             "Machine learning is awesome",
             "NLP is the future!",
             "The children are our future",
             "Our children's children are our grandchildren",
             "The cake was not very good",
             "No one said that the cake was not very good",
             "Life is beautiful"]
vocab = ["awesome", "learning", "children", "cake", "good", "none", "machine"]
E, F = tf_idf(sentences, vocab)
print(E)
print(F)

# Expected Output:
# [[1.         0.         0.         0.         0.         0.
#   0.        ]
#  [0.5098139  0.60831315 0.         0.         0.         0.
#   0.60831315]
#  [0.         0.         0.         0.         0.         0.
#   0.        ]
#  [0.         0.         1.         0.         0.         0.
#   0.        ]
#  [0.         0.         1.         0.         0.         0.
#   0.        ]
#  [0.         0.         0.         0.70710678 0.70710678 0.
#   0.        ]
#  [0.         0.         0.         0.70710678 0.70710678 0.
#   0.        ]
#  [0.         0.         0.         0.         0.         0.
#   0.        ]]
# ['awesome', 'learning', 'children', 'cake', 'good', 'none', 'machine']