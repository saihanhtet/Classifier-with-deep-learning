from flask import Flask, render_template, request
from classify import Classifier

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    context = {'result': ''}
    if request.method == "POST":
        name = request.form['name']
        type = request.form.get('select')
        obj = Classifier(str(type))
        res = obj.predict(name)[0]
        if res == 'M':
            res = "male"
        if res == 'F':
            res = "female"

        res = f"'{name}' is a {res}"
        context = {'result': res}
        return render_template('index.html', context=context)
    else:
        return render_template('index.html', context=context)


if __name__ == '__main__':
    import webbrowser  # import webbrowser package to open in browser
    # open the website in browser with link
    webbrowser.open('http://127.0.0.1:5000/')
    # debug should be false to open with webbrowser and debug = True run twice
    app.run(debug=False)
