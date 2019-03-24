import pandas as pd
import numpy as np
import collections
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib.pyplot as plt


class Utils:
    def __init__(self):
        self.stopwords = ['a', 'di', 'in', 'il', 'per', 'la', 'e', 'i', 'da', '–', 'non', 'un', 'è',
                          'che', 'al', 'con', 'ai', 'le', 'ha', 'se', 'gli', 'degli', 'del', 'ma', 'lo', 'd',
                          'sono', 'una', 'dei', 'della', 'si', 'ci', 'non', 'uccisa', 'tutti', 'questi', 'dall',
                          'l', 'alla', 'su', 'gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno',
                          'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre', '2018', '2019',
                          'dell', 'nel', 'hanno', 'più', 'anche', 'ad', 'come', 'ti', 'dal', 'fa', 'li', 'perché',
                          'all', 'poi', 'dà', 'ho', 'già', 'ne', 'dai', 'sui', 'cui', 'alle', '(ansa)',
                          'aad','aago','aahrus', 'if', 'delle', 'tra', 'function', 'sul', 'ed', 'nella', 'return',
                          'else','cookieaccepted','dopo','for', 'ie', 'o', 'nei', 'googletag', 'detto', 'prima',
                          'codiciaree', 'dalla', 's', 'questo', 'dove', 'questa','uno','lt','suo', 'agli', 'g', 'sulla',
                          'end','nelle', 'homepage', 'itemscope', 'senza', 'start', 'useraccept', 'fra',
                          'sta', 'typeof', 'mi','dagli', 'css', 'elements', 'fav', 'buttons', 'bxslider',
                          'dotnadasyncparamsad', 'polyfill', 'enquire', 'navbar', 'cookie', 'queste', 'var', 'dot',
                          'ansa', 'tag', 'bottom', 'data', 'used', 'true', 'card', 'try', 'and', 'open']
        f = open('data/stopwords.txt', "r")
        self.stopwords += (self.remove_commas(f.read())).split()
        for i in range(32):
            self.stopwords.append(str(i))
        self.corpuses = []
        self.list_of_figures = []
        print('Class has been initialized')

    def remove_commas(self, string):
        string = string.lower()
        string = ' '.join(filter(str.isalpha, string.split()))
        string = string.replace("ã", " ")
        string = string.replace("ã", " ")
        string = string.replace("â", " ")
        string = string.replace("ª", " ")
        return string

    def get_text_from_df(self, df):
        string = ''
        for index, row in df.iterrows():
            string += self.remove_commas(row['article_body'])
        return string

    def get_cardinality(self, df):
        string = ''
        final_string = []
        for index, row in df.iterrows():
            string += self.remove_commas(row['article_body'])

        string = string.split()
        for i in string:
            if i not in self.stopwords:
                final_string.append(i)
        # print(len(final_string))

        counter = collections.Counter(final_string)
        # print(counter.most_common(500))

        # final_string = np.unique(final_string)

        return final_string, counter

    def get_ratio_per_word(self, word, num, counter):
        # computes the frequency of a word in corpus
        try:
            return counter.get(word)/num
        except TypeError:
            # print("Fail!", word, num)
            return 0

    def get_cardinality_book(self, text, description):
        string = self.remove_commas(text)
        final_string = []

        string = string.split()
        for i in string:
            if i not in self.stopwords:
                final_string.append(i)
        print('Length of ' + description + str(len(final_string)))

        counter = collections.Counter(final_string)
        # print(counter.most_common(500))

        # final_string = np.unique(final_string)

        return final_string, counter

    def set_corpuses(self, corpuses):
        self.corpuses = corpuses

    def compute_idf(self, word):
        df = 0
        for i in self.corpuses:
            if word in i:
                df += 1
        return df

    def print_figure(self):
        with PdfPages('output_little.pdf') as pdf:
            for i in self.list_of_figures:
                plt.figure(figsize=(16, 9))
                pdf.savefig(i)  # saves the current figure into a pdf page
                plt.close()
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Tf-idf, cross-entropy'
            d['Author'] = 'Alessandro Artoni'
            d['Subject'] = 'Statistics on a twitter datasets'
            d['Keywords'] = 'Twitter misinformation datasets'
            d['CreationDate'] = datetime.datetime(2019, 3, 14)
            d['ModDate'] = datetime.datetime.today()

    def append_figure(self, figure):
        self.list_of_figures.append(figure)
