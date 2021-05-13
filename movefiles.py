
import csv
import shutil

filenames = []
with open('actual_via_project_csv2.csv', newline='') as csv_file:
    data = csv.reader(csv_file)
    for row in data:
        if row[6]=='{}':
            filenames.append(row[0])
    filenames.pop(0)
print(len(filenames))
for name in filenames:
    try:
        shutil.move('E:/input/' + name, 'E:/data/class2')
    except:
        pass