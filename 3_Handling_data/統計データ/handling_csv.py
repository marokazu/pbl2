import csv

f = open("iris.csv", "r")
reader = csv.reader(f)

y_name = "category"
x_name = []
i = 0
y_id = 0
tmp = []
x = []
y = []
for row in reader:
    if i == 0:#カラムの読み込み
        for j in range(len(row)):
            if row[j] == y_name:
                y_id = j
            elif row[j] != y_name:
                x_name.append(row[j])
        i=1
    elif i != 0:#データの読み込み
        for j in range(len(row)):
            if j == y_id:
                y.append(int(row[j]))
            elif j != y_id:
                tmp.append(float(row[j]))
        x.append(tmp)
        tmp = []


#それぞれの変数をprintしてもらう
