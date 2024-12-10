from flask import Flask, request, render_template, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render a simple HTML page

@app.route('/run-command', methods=['POST'])
def run_command():
    try:
        # Run the Python script command
        process = subprocess.run(
            ['python', 'image_singlepose_heatmap.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Capture the output and return it
        output = process.stdout
        error = process.stderr
        if process.returncode == 0:
            return jsonify({"success": True, "output": output})
        else:
            return jsonify({"success": False, "output": output, "error": error})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
