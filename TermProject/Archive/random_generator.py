import random

with open('csvfile.csv','wb') as file:
    file.write("orderItemID".encode())
    file.write(",".encode())
    file.write("returnShipment".encode())
    file.write('\n'.encode())
    for line in range(50078):
        b = random.randrange(0, 2)
        file.write(str(line+1).encode())
        file.write(",".encode())
        file.write(str(b).encode())
        file.write('\n'.encode())
