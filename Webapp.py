# !/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse

from flask import Flask, render_template, request, send_file, Response
from flask_session import Session
from flask_bootstrap import Bootstrap
from forms import InterfaceForm
from PIL import Image
import io
import glob
import os
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

frame_last = None
frame_w = 0
frame_h = 0
set_roi = False


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    global people_counter_process, people_counter_command_queue, people_counter_output_queue, people_counter_object, frame_h, frame_w, set_roi
    form = InterfaceForm()
    webdata = dict(request.form)

    if people_counter_process.is_alive():
        people_counter_started = True
    else:
        people_counter_started = False
        set_roi = False
        frame_last = None

    # When a button gets pressed:
    if 'submit' in webdata.keys():
        if webdata.get('submit') == 'Start':
            # Get input parameter
            prototxt = webdata.get('prototxt')
            model = webdata.get('model')
            if ('input' in webdata.keys()):
                input = webdata.get('input')
                if input == "":
                    input = None
            else:
                None
            if ('output' in webdata.keys()):
                output = webdata.get('output')
                if output == "":
                    output = None
            else:
                None
            confidence = float(webdata.get(
                'confidence')) if 'confidence' in webdata.keys() else 0.4
            skip_frames = int(webdata.get(
                'skip_frames')) if 'skip_frames' in webdata.keys() else 30
            frame_counts_up = int(webdata.get(
                'frame_counts_up')) if 'frame_counts_up' in webdata.keys() else 8
            orientation = int(webdata.get(
                'orientation')) if 'orientation' in webdata.keys() else 1
            offset_dist = int(webdata.get(
                'offset_dist')) if 'offset_dist' in webdata.keys() else 2
            border_dist = int(webdata.get(
                'border_dist')) if 'border_dist' in webdata.keys() else 50
            roi = True if 'roi' in webdata.keys() else False
            queue = True if 'queue' in webdata.keys() else False

            # Create and init people counter variables
            people_counter_command_queue = Queue()
            people_counter_output_queue = Queue()
            people_counter_object = PeopleCounter(prototxt=prototxt,
                                                  model=model,
                                                  input=input,
                                                  output=output,
                                                  confidence=confidence,
                                                  skip_frames=skip_frames,
                                                  frame_counts_up=frame_counts_up,
                                                  orientation=orientation,
                                                  offset_dist=offset_dist,
                                                  border_dist=border_dist,
                                                  queue=queue,
                                                  roi=roi,
                                                  webserver=True)
            people_counter_process = Process(
                target=people_counter_object.main_loop,
                args=(people_counter_command_queue, people_counter_output_queue,))

            # Start people counter
            people_counter_process.start()
            people_counter_started = True
            set_roi = roi

        elif webdata.get('submit') == 'Stop':
            people_counter_process.terminate()
            people_counter_started = False
            set_roi = False
            frame_last = None
            people_counter_command_queue = Queue()
            people_counter_output_queue = Queue()
            people_counter_process = Process()

        elif webdata.get('submit') == 'Set ROI':
            # (2.0, 2.0, 1078.0, 1918.0)
            x = float(webdata.get('sliderXfrom'))
            x_width = float(webdata.get('sliderXto')) - x
            y = float(webdata.get('sliderYfrom'))
            y_height = float(webdata.get('sliderYto')) - y

            set_roi = False

            people_counter_command_queue.put((x, y, x_width, y_height))
            people_counter_command_queue.put("frame")
            image()

        elif webdata.get('submit') == 'Download CSV':
            return getPlotCSV()

        elif webdata.get('submit') == 'Get current frame':
            people_counter_command_queue.put("frame")
            image()

        elif webdata.get('submit') == 'Debug':
            print("debug")
            print(webdata)
            print(request.form["slider1"])

    return render_template('home.html', form=form, people_counter_started=people_counter_started, frame_h=frame_h, frame_w=frame_w, set_roi=set_roi)


@app.route('/image.png')
def image():
    global people_counter_output_queue, people_counter_command_queue, frame_h, frame_w, frame_last

    try:
        img = people_counter_output_queue.get(timeout=0.5)
        print("Got an output!")

        frame_last = img

        frame_h = img.shape[0]
        print(img.shape[0])
        frame_w = img.shape[1]
        print(img.shape[1])

        img = Image.fromarray(img.astype('uint8'))
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)

        return send_file(file_object, mimetype='image/PNG')
    except Exception as e:
        print(e)
        if frame_last is not None:
            img = Image.fromarray(frame_last.astype('uint8'))
            file_object = io.BytesIO()
            img.save(file_object, 'PNG')
            file_object.seek(0)

            return send_file(file_object, mimetype='image/PNG')
        else:
            return None


def getPlotCSV():
    try:
        list_of_files = glob.glob('csv/*.csv')
        latest_file = max(list_of_files, key=os.path.getctime)

        return send_file(os.path.join(latest_file), as_attachment=True)

    except Exception as e:
        print("Failed to get latest CSV file")
        print(e)

        return None


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
