import csv
from datetime import datetime


def create_csv_file():

    now = datetime.now()

    # Building the Date and Time string
    current_time = now.strftime("%Y_%m_%d_%H-%M-%S")

    # Building the complete filename
    csv_filename = "csv/{}_output.csv".format(current_time)

    # Open File at specified location

    csv_file = open(csv_filename, 'w')

    # Create the csv writer
    writer = csv.writer(csv_file, dialect='excel')

    # Write a row to the file

    writer.writerow(['Date', 'Time', 'Direction', 'Total'])

    # Return the used filename and path

    return csv_filename


def write_new_value(file_name, direction, newtotal):

    now = datetime.now()

    # Building the Date and Time string
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")

    # Open File at specified location

    csv_file = open(file_name, 'a', newline='')

    # Create the csv writer
    writer = csv.writer(csv_file, dialect='excel')

    # Create list with columns as members
    newrow = [current_date, current_time,direction, newtotal]
    # Write a row to the file

    writer.writerow(newrow)
