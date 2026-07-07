from flask import (Flask, render_template, request, redirect, url_for)

app = Flask(__name__, template_folder="mock_ups/html")
app.secret_key = "supersecretkey"

@app.route("/run_fail", methods =["GET"])
def run_fail():
   return render_template("run_fail.html")

@app.route("/run_pass_one", methods =["GET"])
def run_pass_one():
   return render_template("run_pass_option_one.html")

@app.route("/run_pass_two", methods =["GET"])
def run_pass_two():
   return render_template("run_pass_option_two.html")

if __name__ =="__main__":
    app.run(debug=True)