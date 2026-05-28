import pickle
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, session, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Simple user storage (in production, use a database)
users = {}

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    if 'username' in session:
        return send_file("index.html")
    return redirect(url_for('signin'))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists')
            return redirect(url_for('signup'))
        users[username] = password
        session['username'] = username
        return redirect(url_for('home'))
    return render_template("signup.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        flash('Invalid credentials')
        return redirect(url_for('signin'))
    return render_template("signin.html")

@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for('signin'))

@app.route("/predict", methods=["POST"])
def predict():
    if 'username' not in session:
        return jsonify({"error": "Not logged in"}), 401
    text = request.json["text"]

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]

    result = "FAKE" if prediction == 0 else "REAL"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)