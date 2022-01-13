# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import SubmitField, StringField, DecimalField
from wtforms.fields.simple import BooleanField


class InterfaceForm(FlaskForm):
    prototxt = StringField('')
    model = StringField('')
    input = StringField('')
    output = StringField('')
    confidence = DecimalField('')
    skip_frames = DecimalField('')
    frame_counts_up = DecimalField('')
    orientation = DecimalField('')
    offset_dist = DecimalField('')
    border_dist = DecimalField('')
    roi = BooleanField('')
    queue = BooleanField('')

    submit = SubmitField('')
