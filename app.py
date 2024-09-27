from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/profile')
def profile():
    return render_template('profile.html')


@app.route('/summarize')
def summarize():
    return render_template('summarize.html')

@app.route('/extract')
def extract():
    return render_template('extract.html')


if __name__ == '__main__':
    app.run(debug=True)
