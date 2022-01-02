# !/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse

from flask import Flask, render_template, request
from flask_session import Session
from flask_bootstrap import Bootstrap
from forms import InterfaceForm
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
people_counter_queue = Queue()
people_counter_process = Process()
people_counter_started = False


@app.route("/", methods=['GET', 'POST'])
@app.route("/home", methods=['GET', 'POST'])
def home():
    form = InterfaceForm()

    if request.method == 'GET':
        pass
    elif request.method == 'POST':
        if "run_people_counting" in request.form and request.form['run_people_counting'] == 'Start People Counting':
            print("Start counting")

            people_counter_queue = Queue()
            people_counter_object = PeopleCounter("mobilenet_ssd/MBSSD_PED_deploy.prototxt",
                                                  "mobilenet_ssd/MBSSD_PED.caffemodel", input="videos/MOT20-02.webm")
            people_counter_process = Process(
                target=people_counter_object.main_loop, args=(False,))
            people_counter_process.start()
            people_counter_started = True

    return render_template('home.html', form=form)


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
