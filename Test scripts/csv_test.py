from write_csv import create_csv_file
from write_csv import write_new_value

current_file = create_csv_file()
print(current_file)
write_new_value(current_file, 'up', 3)
write_new_value(current_file, 'up', 2)
write_new_value(current_file, 'up', 1)
