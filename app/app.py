from flask import Flask, render_template, render_template_string, request
from similarity import get_most_similar
import pickle

app = Flask(__name__)

with open('embeddings/glove_embeds.pickle', 'rb') as f:
    embeds = pickle.load(f)

@app.route('/', methods=['POST', 'GET'])
def index():
        
    if request.method == 'POST':
        query = request.form['query']
        most_sim = get_most_similar(query, embeds, 10)
        print(most_sim)
        return render_template('home.html', most_sim=most_sim, show="table")
    else:
        return render_template('home.html', show="none")

if __name__ == '__main__':
    app.run(debug=True)