# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import SubmitField, BooleanField, DecimalField


class InterfaceForm(FlaskForm):
    run_people_counting = SubmitField("")
    get_current_frame = SubmitField("")
    download_csv = SubmitField("")
