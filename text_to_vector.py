import sys
sys.path.append('..')
import numpy as np

# text 전처리
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    text = text.replace(',','')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

# 동시발생 행렬 생성함수
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
       for i in range(1,window_size+1):
            left_idx = idx-i
            right_idx = idx+i

            if(left_idx >= 0):
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if (right_idx < corpus_size):
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

# 벡터유사도 리턴 함수
def cos_similarity(x,y,eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2))+eps)
    ny = y / (np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx,ny)

# 단어->벡터 / 해당벡터와 유사도높은 벡터 순서대로 출력
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 검색어를 꺼낸다.
    if query not in word_to_id:
        print('%s를 찾을 수 없습니다.' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i],query_vec)

    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0

    for i in (-1* similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count+=1
        if count >= top:
            return

text = "Sorry, honey. Let's see. we're meeting the Sombergs for lunch and four of us decided to try a new restaurant." \
       "In the afternoon, your dad and I plan to visit the Museum of Broadcasting and maybe walk around Lincoln Center." \
       " Then we're going to meet the Wileys for dinner, and after that we're going to see 'The Phantom of the Opera'." \
       " Then we'll come home."

corpus, word_to_id , id_to_word = preprocess(text)
# print(corpus)
# print(word_to_id)
# print(id_to_word)

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['afternoon']]
c1 = C[word_to_id['visit']]

# c0과 c1의 코사인유사도 출력
print(cos_similarity(c0,c1,1e-8))

# 'afternoon'과 유사도높은 순서대로 출력
print(most_similar('afternoon',word_to_id,id_to_word,C))

