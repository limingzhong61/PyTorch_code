sheet_a_name = 'A1'
getname = []
for a in sheet_a_name:
    getname.append(a)
filesheet = 'a1'
sheeta = []
for a in filesheet:
    sheeta.append(a)


def is_lower(c):
    return 'a' <= c <= 'z'


for b in range(0, len(filesheet)):
    if is_lower(getname[b]):
        getname[b] = getname[b] - 'a' + 'A'
    if is_lower(sheeta[b]):
        sheeta[b] =int(sheeta[b]) - 'a' + 'A'

    print(getname[b], sheeta[b])
