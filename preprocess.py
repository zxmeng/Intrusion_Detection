import numpy as np
from numpy import genfromtxt
from numpy import savetxt


print "loading data..."
# dataset = genfromtxt("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", delimiter=',', dtype=None, names=True)

# Monday-WorkingHours.pcap_ISCX.csv
# Tuesday-WorkingHours.pcap_ISCX.csv
# Wednesday-workingHours.pcap_ISCX.csv
# Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
# Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
# Friday-WorkingHours-Morning.pcap_ISCX.csv
# Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
# Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv

dataset = open("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
content = dataset.readlines()

row = len(content)
selected = np.zeros((row-1, 7)).astype(float)

i = 0
for line in content[1:]:
	tokens = line.split(",")
	selected[i,0] = tokens[7]
	selected[i,1] = tokens[10]
	selected[i,2] = tokens[17]
	selected[i,3] = tokens[19]
	selected[i,4] = tokens[30]
	selected[i,5] = tokens[69]

	if "BENIGN" in tokens[84]:
		selected[i,6] = 0
	else:
		selected[i,6] = 1

	i+=1

np.savetxt('train_fri_3.csv', selected[1:,:], fmt='%1.2f', delimiter=',')

