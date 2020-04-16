#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main():
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('file_name', type=str,
	                    help='name of the thermal log file')
	parser.add_argument('--thermal_threshold', type=int, default=0,
	                    help='Threshold for the card to throttle')

	args = parser.parse_args()
	
	file_name = args.file_name
	thermal_threshold = args.thermal_threshold

	t = []
	throughput = []
	
	# second, throughput, temp[, temp[, temp...]]
	with open(file_name) as f:
		for line in f:
			iterms = line.split(', ')
			num_gpu = len(iterms) - 3
			break

	temperature = []
	for i in range(num_gpu):
		temperature.append([])

	with open(file_name) as f:
		for line in f:
			iterms = line.split(', ')
			print(iterms)
			t.append(int(iterms[0]))
			throughput.append(float(iterms[1]))
			for i in range(2, len(iterms) - 1):
				temperature[i-2].append(int(iterms[i]))

	t = np.asarray(t)
	throughput = np.asarray(throughput)
	temperature = np.asarray(temperature)

	fig, (ax0, ax1) = plt.subplots(2, 1)
	fig.suptitle(file_name[:-4], fontsize=8)

	ax0.set_xlim((t[0], t[-1]))
	ax0.set_ylim((0, np.amax(throughput) * 1.1))

	ax0.set_title('Throughput')
	ax0.set_xlabel('Time (sec)')
	ax0.set_ylabel('Throughput (images/sec)')	
	ax0.plot(t, throughput, 'b')

	ax1.set_xlim((t[0], t[-1]))
	ax1.set_ylim((0, np.amax(np.amax(temperature)) * 1.25))
	ax1.set_title('temperature')
	ax1.set_xlabel('Time (sec)')
	ax1.set_ylabel('Degree (celsius)')		
	for i in range(num_gpu):
		ax1.plot(t, temperature[i], 'b')
	
	if thermal_threshold > 0:
		ax1.plot(t, thermal_threshold * np.ones_like(t), 'r')

	plt.show()


if __name__ == '__main__':
	main()
