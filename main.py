import pandas as pd
from utils import Utils
import collections
from math import log
import numpy as np


#################################################################################
#                               Data pre-processing                             #
#################################################################################

# fake news article
df = pd.read_pickle('data/article_partial.pkl')
# df = df.rename(columns={'content':'article_body'})

# articles
df2 = pd.read_pickle('data/ansa.pkl')

# Documents used for tf-idf
f = open('data/il_re_bello.txt', "r")
f2 = open('data/una_donna.txt', "r")
f3 = open('data/storia_d_italia.txt','r')

book1 = f.read()
book2 = f2.read()
book3 = f3.read()

book = book1 + book2 + book3

util = Utils()

text_fake_news = util.get_text_from_df(df)
text_articles = util.get_text_from_df(df2)
corpuses = [book1, book2, book3, text_fake_news, text_articles]

util.set_corpuses(corpuses)

# v is the fake news
vocabulary_v, counter_v = util.get_cardinality_book(text_fake_news, ' fake news ')
# u are the articles  / the other corpus
# vocabulary_u, counter_u = util.get_cardinality(df2)
vocabulary_u, counter_u = util.get_cardinality_book(text_articles, ' articles ')

vocabulary_books, counter_books = util.get_cardinality_book(book, ' book total ')

vocabulary_b1, counter_b1 = util.get_cardinality_book(book1, ' book1 ')
vocabulary_b2, counter_b2 = util.get_cardinality_book(book2, ' book2 ')
vocabulary_b3, counter_b3 = util.get_cardinality_book(book3, ' book3 ')


p_x = []
q_x = []
most_common_words = []

# select only some words for time reasons
for i in counter_u.most_common(500):
    most_common_words.append(i[0])

for j in counter_v.most_common(500):
    most_common_words.append(j[0])

most_common_words = np.unique(most_common_words)

#################################################################################
#                       Computing tf-idf, cross entropy                         #
#################################################################################


tf_idf = {'word': [], 'fake_news': [], 'articles': [], 'book1': [], 'book2': [], 'book3': []}

for i in most_common_words:
    freq_fake_news = util.get_ratio_per_word(i, len(vocabulary_v), counter_v)
    freq_articles = util.get_ratio_per_word(i, len(vocabulary_u), counter_u)
    # freq_book = util.get_ratio_per_word(i, len(vocabulary_books), counter_books)
    freq_book1 = util.get_ratio_per_word(i, len(vocabulary_b1), counter_b1)
    freq_book2 = util.get_ratio_per_word(i, len(vocabulary_b2), counter_b2)
    freq_book3 = util.get_ratio_per_word(i, len(vocabulary_b3), counter_b3)

    p_x.append(freq_articles)
    q_x.append(freq_fake_news)

    tf_idf['word'].append(i)
    tf_idf['fake_news'].append(freq_fake_news*util.compute_idf(i))
    tf_idf['articles'].append(freq_articles * util.compute_idf(i))
    tf_idf['book1'].append(freq_book1*util.compute_idf(i))
    tf_idf['book2'].append(freq_book2*util.compute_idf(i))
    tf_idf['book3'].append(freq_book3 * util.compute_idf(i))
    # tf_idf['book'].append(freq_book * util.compute_idf(i))

h_p_q = 0
h_p_q_x = []


for i in range(len(p_x)):
    try:
        el = -(p_x[i]*log(q_x[i]))
    except ValueError:
        el = 0
    h_p_q_x.append(el)
    h_p_q += el

#################################################################################
#                              Plotting results                                 #
#################################################################################

array = np.array(h_p_q_x)
most_common_words = np.array(most_common_words)
y = np.unravel_index(np.argsort(h_p_q_x, axis=None), array.shape)

best_words = most_common_words[y][-15:]
best_words_values = array[y][-15:]

words_contribution = pd.DataFrame({'word': best_words, 'contribution': best_words_values})
words_contribution = words_contribution.set_index('word')
plotted = words_contribution.plot(kind='bar', figsize=(16, 9))
plotted.set(title='Word contribution to cross entropy')
util.append_figure(plotted.get_figure())

df = pd.DataFrame(tf_idf)


df.to_pickle('graph.pkl')

df = df.set_index('word')


for i in list(df):
    plotted = df.sort_values(by=i)[-10:].plot(kind='bar', figsize=(16, 9))
    plotted.set(title='Tf-idf ordered by ' + i)
    util.append_figure(plotted.get_figure())

df = df[['articles', 'fake_news']]

for i in list(df):
    plotted = df.sort_values(by=i)[-10:].plot(kind='bar', figsize=(16, 9))
    plotted.set(title='Tf-idf ordered by ' + i)
    util.append_figure(plotted.get_figure())

util.print_figure()
