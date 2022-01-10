# !/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse

from flask import Flask, render_template, request, send_file, Response
from flask_session import Session
from flask_bootstrap import Bootstrap
from forms import InterfaceForm
from PIL import Image
import io
from multiprocessing import Process, Queue

from people_counter import PeopleCounter

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

app.config['BOOTSTRAP_SERVE_LOCAL'] = True
Bootstrap(app)

sessInst = Session()
sessInst.init_app(app)

people_counter_object = None
people_counter_command_queue = Queue()
people_counter_output_queue = Queue()
people_counter_process = Process()


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    global people_counter_process, people_counter_command_queue, people_counter_output_queue, people_counter_object
    form = InterfaceForm()
    if people_counter_process.is_alive():
        people_counter_started = True
    else:
        people_counter_started = False

    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        if "run_people_counting" in request.form and request.form['run_people_counting'] == 'Start People Counting':
            print("Start counting")

            people_counter_command_queue = Queue()
            people_counter_output_queue = Queue()
            people_counter_object = PeopleCounter(prototxt="mobilenet_ssd/MobileNetSSD_deploy_py.prototxt",
                                                  model="mobilenet_ssd/MobileNetSSD_deploy_py.caffemodel", input="videos/test_video.mp4", skip_frames=10, queue=False, roi=True)
            people_counter_process = Process(
                target=people_counter_object.main_loop, args=(people_counter_command_queue, people_counter_output_queue,))
            people_counter_process.start()
            people_counter_started = True

        elif "run_people_counting" in request.form and request.form['run_people_counting'] == 'Stop People Counting':
            print("Stop counting")
            people_counter_command_queue = Queue()
            people_counter_output_queue = Queue()
            people_counter_process.kill()
            people_counter_process = Process()
            people_counter_started = False

        elif "download_csv" in request.form and request.form['download_csv'] == 'Download CSV':
            return getPlotCSV()

        elif "get_current_frame" in request.form and request.form['get_current_frame'] == 'Get current frame':
            image()

    return render_template('home.html', form=form, people_counter_started=people_counter_started)


@app.route('/image.png')
def image():
    global people_counter_output_queue, people_counter_command_queue
    print("Get output")
    people_counter_command_queue.put(1)

    try:
        img = people_counter_output_queue.get(timeout=1)
        print("Got an output!")

        img = Image.fromarray(img.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)
        print(type(img))

        return send_file(file_object, mimetype='image/PNG')
    except:
        print("No new output")
        return None


def getPlotCSV():
    with open("2022_01_04_16-28-24_output.csv") as fp:
        csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=file.csv"})


def main():
    # Start webserver

    parser = argparse.ArgumentParser(description='Nope ...')
    parser.add_argument(
        '-d', '--debug', action='store_true', help='Nope ...')
    args = parser.parse_args()
    print(f'Starting webapp with Debug = {args.debug}')

    if args.debug:
        app.run(host='0.0.0.0', debug=args.debug,
                use_reloader=args.debug)
    else:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
