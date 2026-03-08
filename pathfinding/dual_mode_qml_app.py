from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

from dual_mode_qml_backend import TOTAL_CASES, run_mode

APP_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(APP_ROOT / 'templates'))


@app.get('/')
def index():
    return render_template('dual_mode_qml.html', case_count=TOTAL_CASES)


@app.get('/api/run')
def api_run():
    mode = request.args.get('mode', 'mapping')
    seed = int(request.args.get('seed', 11))
    case_index = max(0, min(TOTAL_CASES - 1, int(request.args.get('case', 0))))
    if mode not in ('mapping', 'path'):
        return jsonify({'error': 'mode must be mapping or path'}), 400
    return jsonify(run_mode(mode=mode, seed=seed, case_index=case_index))


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8030, debug=False, use_reloader=False)
