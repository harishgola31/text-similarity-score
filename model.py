import nltk
import pickle
from helper import cleaning, averaged_word_vectorizer, cosine_similarity


def similarity(data):

    nltk.download('stopwords')
    nltk.download('wordnet')
    
    data['cleaned_text1'] = data.text1.apply(lambda x: cleaning(x))
    data['cleaned_text2'] = data.text2.apply(lambda x: cleaning(x))
    data['cleaned_text1'] = data.cleaned_text1.apply(lambda x: nltk.word_tokenize(x))
    data['cleaned_text2'] = data.cleaned_text2.apply(lambda x: nltk.word_tokenize(x))

    # Loading pre trained model for word embedding
    with open('word_embedded.pkl', 'rb') as file:
        model = pickle.load(file) 
    feature_size = model.vector_size

    # get document level embeddings
    ft_text1_features = averaged_word_vectorizer(corpus=data.cleaned_text1.to_list(), model=model, num_features=feature_size)
    ft_text2_features = averaged_word_vectorizer(corpus=data.cleaned_text2.to_list(), model=model, num_features=feature_size)

    data['text1_vec'] = [i for i in ft_text1_features]
    data['text2_vec'] = [i for i in ft_text2_features]

    simi_results = []
    for i in range(data.shape[0]):
        simi_results.append(cosine_similarity(data.text1_vec.iloc[i], data.text2_vec.iloc[i]))

    data['cos_similarity'] = simi_results

    return data
