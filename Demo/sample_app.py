from flask import Flask
app = Flask(__name__)

@app.route('/sample')
def runing():
	return "FIRST PAGE"

if __name__=="__main__":
	app.run(debug=True)