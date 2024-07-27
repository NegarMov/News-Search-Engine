import json
import string
from hazm import *
import pickle
import os.path
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
import time

def convert(text):
    reshaped_text = arabic_reshaper.reshape(text)
    converted = get_display(reshaped_text)
    return converted

# initate the normalizer, stemmer and lemmatizer
normalizer = Normalizer()
lemmatizer = Lemmatizer()

# create a list of stopwords and punctuations
stopwords = stopwords_list()
punctuations = list(string.punctuation) + ['،', '؛', '»', '«', '؟']

RESULT_SIZE = 5
CHAMPION_SIZE = 50

class Document:
    def __init__(self, id, title, content, url):
        self.id = id
        self.title = title
        self.content = content
        self.url = url
        self.total_terms = 0

class Token:
    def __init__(self, doc_id, term, position):
        self.doc_id = doc_id
        self.term = term
        self.position = position

    def __str__(self):
        return f"{self.doc_id} - {self.position} : {self.term}"

    def __repr__(self):
        return self.__str__()

class Postings:
    def __init__(self, doc_id, term_frequency=0, tfidf = 0):
        self.doc_id = doc_id
        self.term_frequency = term_frequency
        self.positions = []
        self.tfidf = tfidf

    def add_posting(self, position):
        self.positions.append(position)
        self.term_frequency += 1

    def __str__(self):
        return f"{self.doc_id} ({self.tfidf}) : #{self.term_frequency} -> {self.positions}"

    def __repr__(self):
        return self.__str__()

class Index:
    def __init__(self):
        self.doc_frequency = 0
        self.postings_list = []
        self.champoin_list = []

    def add_postings(self, postings):
        self.postings_list.append(postings)
        self.doc_frequency += 1

    def __str__(self):
        return f"#{self.doc_frequency} -> {self.postings_list}"

    def __repr__(self):
        return self.__str__()

class Query:
    def __init__(self, query):
        self.query = query
        self.tokens = []
        self.not_tokens = []
        self.parse_query()

    def parse_query(self):
        q = normalizer.normalize(self.query).replace('!', ' !').split(' ')
        i = 0
        is_not = False

        while i < len(q):
            if q[i] == '!':
                is_not = True
                i += 1
            else:
                token = q[i]
                if q[i].startswith('"') or q[i].startswith('«'):
                    token = q[i][1:]
                    i += 1
                    while not (q[i].endswith('"') or q[i].endswith('»')):
                        token += " " + q[i]
                        i += 1
                    token += " " + q[i][:-1]

                    edited_token = ""
                    for word in token.split(' '):
                        lemmatized_word = lemmatizer.lemmatize(word)
                        if (lemmatized_word not in stopwords) and (lemmatized_word not in punctuations) and (word not in stopwords) and (word not in punctuations):
                            edited_token += lemmatized_word + " "
                    token = edited_token.strip()
                
                lemmatized_token = lemmatizer.lemmatize(token)

                if (lemmatized_token not in stopwords) and (lemmatized_token not in punctuations) and (token not in stopwords) and (token not in punctuations):
                    if is_not:
                        self.not_tokens.append(lemmatized_token)
                        is_not = False
                    else:
                        self.tokens.append(lemmatized_token)
                i += 1

    def get_combinations(self):
        def sub_lists(l):
            lists = [[]]
            for i in range(len(l) + 1):
                for j in range(i):
                    lists.append(l[j: i])
            return lists

        token_combinations = [(x, y) for x in sub_lists(self.tokens) for y in sub_lists(self.not_tokens)]
        return sorted(token_combinations, key=lambda p: 1.01 * len(p[0]) + len(p[1]), reverse=True)[:-1]
    
def merge_postings(postings1, postings2, contains_not=False, distance=-1):
    result = []

    i, j = 0, 0
    while i < len(postings1) and j < len(postings2):
        if postings1[i].doc_id == postings2[j].doc_id:
            if not contains_not:
                if distance == -1:
                    total_frequency = postings1[i].term_frequency + postings2[j].term_frequency
                    result.append(Postings(postings1[i].doc_id, total_frequency))
                else:
                    total_frequency = 0
                    for pos1 in postings1[i].positions:
                        if pos1 + distance in postings2[j].positions:
                            total_frequency += 1
                    if total_frequency > 0:
                        result.append(Postings(postings1[i].doc_id, total_frequency))
            i += 1
            j += 1
        elif postings1[i].doc_id < postings2[j].doc_id:
            if contains_not:
                result.append(Postings(postings1[i].doc_id, postings1[i].term_frequency))
            i += 1
        else:
            j += 1

    if contains_not and i < len(postings1):
        result += postings1[i:]

    return result

def get_postings(token, use_champion = False):
    if ' ' in token: # phrase
        result = []
        phrase_tokens = token.split(' ')

        for i in range(len(phrase_tokens) - 1):
            for j in range(i + 1, len(phrase_tokens)):
                postings1 = positional_index[phrase_tokens[i]].postings_list
                postings2 = positional_index[phrase_tokens[j]].postings_list
                if len(result) == 0:
                    result = merge_postings(postings1, postings2, distance=(j - i))
                else:
                    result = merge_postings(result, merge_postings(postings1, postings2, distance=(j - i)))

        return result
    elif token in positional_index: # single token
        if use_champion:
            return positional_index[token].champion_list
        else:
            return positional_index[token].postings_list
    else: # term not found in dictionary
        return []

def retrieve(query_string):
    query = Query(query_string)
    
    result = []
    for tokens, not_tokens in query.get_combinations():
        res = []

        if len(tokens) == 0 and len(not_tokens) != 0:
            res = allPostings

        for token in tokens:
            if len(res) == 0:
                res = get_postings(token)
            else:
                res = merge_postings(res, get_postings(token))

        for not_token in not_tokens:
            if len(res) == 0:
                break
            else:
                res = merge_postings(res, get_postings(not_token), contains_not=True)

        result += sorted(res, key=lambda p: p.term_frequency, reverse=True)

    print(f"{len(result)} documents found - Top {RESULT_SIZE} search results for: {convert(query_string)}")
    for postings in result[:RESULT_SIZE]:
        print(f"    (#{postings.term_frequency}) {convert(documents[postings.doc_id].title)}")
        print(f"        Document URL: {documents[postings.doc_id].url}")
    print()

    return result

#######################################################################

class RankedQuery:
    def __init__(self, query):
        self.query = query
        self.tokens = []
        self.parse_query()

    def parse_query(self):
        tokens = normalizer.normalize(self.query).split(' ')
        
        tokens_frequency = {}
        for token in tokens:
            lemmatized_token = lemmatizer.lemmatize(token)
            if (lemmatized_token not in stopwords) and (lemmatized_token not in punctuations) and (token not in stopwords) and (token not in punctuations):
                if lemmatized_token in positional_index:
                    if lemmatized_token in tokens_frequency:
                        tokens_frequency[lemmatized_token] += 1
                    else:
                        tokens_frequency[lemmatized_token] = 1

        norm = 0
        for token, frequency in tokens_frequency.items():
            df = positional_index[token].doc_frequency
            tfidf = (1 + frequency) * np.log10(len(documents) / df)
            self.tokens.append((token, tfidf))
            norm += tfidf ** 2
        self.tokens = list(map(lambda t: (t[0], t[1] / np.sqrt(norm)), self.tokens))

def get_cosine_score(postings1, postings2, c2):
    result = []

    i, j = 0, 0
    while i < len(postings1) or j < len(postings2):
        if (i < len(postings1) and j < len(postings2)) and postings1[i].doc_id == postings2[j].doc_id:
            score = postings1[i].tfidf + postings2[j].tfidf * c2
            result.append(Postings(postings1[i].doc_id, tfidf=score))
            i += 1
            j += 1
        elif (i < len(postings1) and not j < len(postings2)) or \
             (i < len(postings1) and j < len(postings2)) and postings1[i].doc_id < postings2[j].doc_id:
            score = postings1[i].tfidf
            result.append(Postings(postings1[i].doc_id, tfidf=score))
            i += 1
        else:
            score = postings2[j].tfidf * c2
            result.append(Postings(postings2[j].doc_id, tfidf=score))
            j += 1

    return result

def ranked_retrieve(query_string, measure, use_champion = False):
    query = RankedQuery(query_string)
    
    if measure == "cosine":
        res = []
        for token, tfidf in query.tokens:
            res = get_cosine_score(res, get_postings(token, use_champion), tfidf)
            
        result = sorted(res, key=lambda p: p.tfidf, reverse=True)

        print(f"{len(result)} documents found - Top {RESULT_SIZE} search results for: {convert(query_string)}")
        for postings in result[:RESULT_SIZE]:
            print(f"    ({postings.tfidf:.4f}) {convert(documents[postings.doc_id].title)}")
            print(f"        Document URL: {documents[postings.doc_id].url}")
        print()
    
    else:
        res = {}
        for token, _ in query.tokens:
            for postings in get_postings(token, use_champion):
                if postings.doc_id in res:
                    res[postings.doc_id] += 1
                else:
                    res[postings.doc_id] = 1
        
        result = []
        for doc_id in res:
            intersect_size = res[doc_id]
            union_size = documents[doc_id].total_terms + len(query.tokens) - intersect_size
            result.append((doc_id, intersect_size / union_size))
        result = sorted(result, key=lambda p: p[1], reverse=True)

        print(f"{len(result)} documents found - Top {RESULT_SIZE} search results for: {convert(query_string)}")
        for doc in result[:RESULT_SIZE]:
            print(f"    ({doc[1]:.4f}) {convert(documents[doc[0]].title)}")
            print(f"        Document URL: {documents[doc[0]].url}")
        print()

#######################################################################

# The main positional index of the engine
positional_index = {}
allPostings = []
documents = []

# check if the index file exists. If not, index the documents
if os.path.exists('./index.pkl'):
    print("Index file exists. Reading it from the file...\n")
    with open ('./index.pkl', 'rb') as index_file:
        positional_index = pickle.load(index_file)
        documents = pickle.load(index_file)
        allPostings = pickle.load(index_file)
else:
    print("Index file does not exist. Indexing the documents...\n")

    # Read the data from the JSON file
    f = open('IR_data_news_12k.json')
    data = json.load(f)
    f.close()

    # create a list of all the documents
    for doc_id in data:
        documents.append(Document(int(doc_id), data[doc_id]['title'], data[doc_id]['content'], data[doc_id]['url']))
        allPostings.append(Postings(int(doc_id), 1))

    # create tokens
    tokens = []
    for doc in documents:
        doc_content = doc.content
        normalized_content = normalizer.normalize(doc_content)
        doc_tokens = word_tokenize(normalized_content)

        # omit the last 2 tokens which mark the end of doc content
        if 'انتهای پیام/' in doc_content:
            doc_tokens = doc_tokens[:-2]
        
        position = 0
        for token in doc_tokens:
            lemmatized_token = lemmatizer.lemmatize(token)
            if (lemmatized_token not in stopwords) and (lemmatized_token not in punctuations) and (token not in stopwords) and (token not in punctuations):
                tokens.append(Token(doc.id, lemmatized_token, position))
                position += 1

    # sort the tokens alphabetically
    tokens.sort(key=lambda x: x.term)

    # print("sorted tokens: ", tokens)
    # print("number of tokens: ", len(tokens))
    # print("\n")

    #######################################################################

    # create the indices
    i = 0
    tfidf_norms = [0] * len(documents)
    while i < len(tokens):
        index = Index()
        last_term = tokens[i].term
        while i < len(tokens) and tokens[i].term == last_term:
            last_id = tokens[i].doc_id
            postings = Postings(last_id)
            while i < len(tokens) and tokens[i].term == last_term and tokens[i].doc_id == last_id:
                postings.add_posting(tokens[i].position)
                i += 1
            documents[last_id].total_terms += 1
            index.add_postings(postings)
        
        for postings in index.postings_list:
            postings.tfidf = (1 + postings.term_frequency) * np.log10(len(documents) / index.doc_frequency)
            tfidf_norms[postings.doc_id] += postings.tfidf ** 2
        
        positional_index[last_term] = index
    tfidf_norms = np.sqrt(tfidf_norms)

    for term in positional_index:
        for postings in positional_index[term].postings_list:
            postings.tfidf /= tfidf_norms[postings.doc_id]
        positional_index[term].champion_list = \
            sorted(positional_index[term].postings_list, key=lambda p: p.tfidf, reverse=True)[:CHAMPION_SIZE]
        positional_index[term].champion_list.sort(key=lambda p: p.doc_id)

    # print("positional index: ", positional_index)
    # print("dictionary size: ", len(positional_index))
    # print("\n")

    with open('./index.pkl', 'wb') as index_file:
        pickle.dump(positional_index, index_file)
        pickle.dump(documents, index_file)
        pickle.dump(allPostings, index_file)

#######################################################################

# retrieve('باشگاه های فوتسال آسیا')

# retrieve('باشگاه های فوتسال ! آسیا')

# retrieve('"سهمیه المپیک"')

# retrieve('طلای "لیگ برتر" ! والیبال')

# retrieve('مایکل ! جردن')

#######################################################################

print("~~~~~~~~~~~ POSITIONAL ~~~~~~~~~~~\n")

# retrieve("باشگاه")

retrieve("سرمربی فوتبال ایران")

# retrieve("رمزارز")

retrieve("تست مثبت کرونا")

print("\n\n")

#######################################################################

print("~~~~~~~~ RANKED - COSINE ~~~~~~~~~\n")
start_time = time.time()

ranked_retrieve("باشگاه", "cosine")

ranked_retrieve("سرمربی فوتبال ایران", "cosine")

ranked_retrieve("رمزارز", "cosine")

ranked_retrieve("تست مثبت کرونا", "cosine")

print("--- %.3f seconds ---" % (time.time() - start_time))
print("\n\n")

#######################################################################

print("~~~~~~~~ RANKED - JACCARD ~~~~~~~~\n")
start_time = time.time()

ranked_retrieve("باشگاه", "jaccard")

ranked_retrieve("سرمربی فوتبال ایران", "jaccard")

ranked_retrieve("رمزارز", "jaccard")

ranked_retrieve("تست مثبت کرونا", "jaccard")

print("--- %.3f seconds ---" % (time.time() - start_time))
print("\n\n")

#######################################################################

print("~~~~~~~ CHAMPION - COSINE ~~~~~~~~\n")
start_time = time.time()

ranked_retrieve("باشگاه", "cosine", use_champion=True)

ranked_retrieve("سرمربی فوتبال ایران", "cosine", use_champion=True)

ranked_retrieve("رمزارز", "cosine", use_champion=True)

ranked_retrieve("تست مثبت کرونا", "cosine", use_champion=True)

print("--- %.3f seconds ---" % (time.time() - start_time))
print("\n\n")
