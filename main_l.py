import pandas as pd

sheet_a_name = 'A1'
getname = []
for a in sheet_a_name:
    getname.append(a)
filesheet = 'a1'
sheeta = []
for a in filesheet:
    sheeta.append(a)
print(getname[0])
print('A' <= getname[0] <= 'Z')
for b in range(0, len(filesheet)):
    if 65 <= ord(getname[b]) <= 122 and 65 <= ord(sheeta[b]) <= 122:
        if getname[b] != sheeta[b]:
            if 65 <= ord(getname[b]) <= 90:
                getname[b] = getname[b].lower()
            elif 97 <= ord(getname[b]) <= 122:
                getname[b] = getname[b].upper()
    print(getname[b], sheeta[b])
