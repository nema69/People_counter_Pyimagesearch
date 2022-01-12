# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, DecimalField
from wtforms.fields.simple import BooleanField


class InterfaceForm(FlaskForm):
    prototxt = StringField("mobilenet")
    model = StringField("model")
    input = StringField("")
    output = StringField("")
    confidence = DecimalField("")
    skip_frames = DecimalField("")
    roi = BooleanField("")
    queue = BooleanField("")

    run_people_counting = SubmitField("")
    get_current_frame = SubmitField("")
    download_csv = SubmitField("")
