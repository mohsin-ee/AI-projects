from flask import Flask, render_template, request
from gensim.models import Word2Vec
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




app = Flask(__name__)


## Loading the necessary data
data = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")


def get_and_sort_corpus(data):
  corpus_sorted = []
  for doc in data['Cleaned-Ingredients'].values:
    doc = sorted(doc.split(sep = ','))
    corpus_sorted.append(doc)
  #print(corpus_sorted)
  return corpus_sorted


class tfidfEmbeddingVectorizer(object):

    def __init__(self, model_cbow):

      self.model_cbow = model_cbow
      self.vector_size = model_cbow.vector_size
      self.word_idf_weight = None

    def fit(self, docs):
    #building a tfidf model to compute each words idf as its weight

      text_docs = []
      for doc in docs:
        text_docs.append(" ".join(doc))

      tfidf = TfidfVectorizer()
      tfidf.fit(text_docs)

      #if a word was never seen before, it is given idf of max of known idf values
      max_idf = max(tfidf.idf_)
      self.word_idf_weight = defaultdict(lambda: max_idf, [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items() ],
                                       )
      return self

    def transform(self, docs):
      doc_word_vector = self.doc_average_list(docs)
      return doc_word_vector

    def doc_average(self, doc):
      # compute the weighted mean of document's word embeddings 
      mean = []
      for word in doc:
        if word in self.model_cbow.wv.index_to_key:
            mean.append(self.model_cbow.wv.get_vector(word) * self.word_idf_weight[word])

      if not mean:
        return np.zeros(self.vector_size)

      else:
        mean = np.array(mean).mean(axis = 0)
        return mean

    def doc_average_list(self,docs):
      return np.vstack([self.doc_average(doc) for doc in docs])
  
  
  
  
def get_recommendations(N, scores):
    """
    Rank scores and output a pandas data frame containing all the details of the top N recipes.
    :param scores: list of cosine similarities
    """
    # load in recipe dataset
    df_recipes = pd.read_csv("Cleaned_Indian_Food_Dataset.csv")
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "url","image-url","TotalTimeInMins","count"])
    count = 0
    for i in top:
        recommendation.at[count, "recipe"] = df_recipes["TranslatedRecipeName"][i]
        recommendation.at[count, "ingredients"] = df_recipes["Cleaned-Ingredients"][i]
        recommendation.at[count, "url"] = df_recipes["URL"][i]
        recommendation.at[count, "image-url"] = df_recipes["image-url"][i]
        recommendation.at[count, "TotalTimeInMins"] = df_recipes["TotalTimeInMins"][i]
        recommendation.at[count, "count"] = df_recipes["Ingredient-count"][i]
        count += 1
    return recommendation

def get_recs(ingredients,mean=False, N=6):
  #load word2vec model
  model = Word2Vec.load("model_cbow.model")

  #normalize embeddings
  model.init_sims(replace = True)
  if model:
    print("Successfully loaded model")
  
  #create corpus
  corpus = get_and_sort_corpus(data)
  

  if mean:

    mean_vec_tr = MeanEmbeddingVectoriser(model)
    doc_vec = mean_vec_tr.transform(corpus)
    doc_vec = [doc.reshape(1,-1) for doc in doc_vec]
    assert len(doc_vec) == len(corpus)

  else:
    tfidf_vec_tr = tfidfEmbeddingVectorizer(model)
    tfidf_vec_tr.fit(corpus)
    doc_vec = tfidf_vec_tr.transform(corpus)
    doc_vec = [doc.reshape(1,-1) for doc in doc_vec]
    assert len(doc_vec) == len(corpus)

  #create embeddings for input
  input =  ingredients
  input = input.split(",")

  if mean:
    input_embedding = mean_vec_tr.transform([input])[0].reshape(1,-1)

  else:
    input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1,-1)

      
      # get cosine similarity between input embedding and all the document embeddings
  cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
  scores = list(cos_sim)
    # Filter top N recommendations
  recommendations = get_recommendations(N, scores)
  return recommendations

def convert_to_hr_min(minutes_array):
    hr_min_array = []
    for minutes in minutes_array:
        hours = minutes // 60
        mins = minutes % 60
        if hours == 0:
            hr_min_array.append(f"{mins} min")
        else:
            hr_min_array.append(f"{hours} hr {mins} min")
    return hr_min_array


def split(data):
   data = data.strip()
   return " ".join(data.split(" ")[:3])


@app.route('/')
def main():
    cards = False
    return render_template("index.html")


@app.route('/dashboard')
def dashboard():
    cards = False
    return render_template("dashboard.html", cards= cards)


@app.route('/generate', methods=['POST', 'GET'])
def generateList():
    if request.method == 'POST':
        print(request.form)
        cards = True
        first_ingredient = request.form['firstIngredient']
        second_ingredient = request.form['secondIngredient']
        third_ingredient = request.form['thirdIngredient']
        fourth_ingredient = request.form['fourthIngredient']
        fifth_ingredient = request.form['fifthIngredient']
        sixth_ingredient = request.form['sixthIngredient']
        
        if not first_ingredient and not second_ingredient and not third_ingredient and not fourth_ingredient and not fifth_ingredient and not sixth_ingredient:
            cards = False
            return render_template("/dashboard.html" ,cards = cards)
        else:
            print("At least one ingredient is not empty.")

            # Combine all ingredients into a single string
            ingredients = ','.join([first_ingredient, second_ingredient, third_ingredient,
                                    fourth_ingredient, fifth_ingredient, sixth_ingredient])
            # print(ingredients)
            # input = "bell pepper ,potato ,tomato"
            rec = get_recs(ingredients)
            # print(rec)
            
            RecipeImageUrl = list(rec["image-url"].values)
            RecipeUrl = list(rec["url"].values)
            RecipeCount = list(rec["count"].values)
            RecipeTime = list(rec["TotalTimeInMins"].values)
            RecipeNames = list(rec["recipe"].apply(split))
            
            ## Converting time to format
            RecipeTime = convert_to_hr_min(RecipeTime)
            
            return render_template("/dashboard.html" ,RIU = RecipeImageUrl ,RU = RecipeUrl ,RC = RecipeCount ,RT = RecipeTime ,RN = RecipeNames ,cards = cards)







if __name__ == "__main__":
    app.run(port = 3000 ,debug =True)