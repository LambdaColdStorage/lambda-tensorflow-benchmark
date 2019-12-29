#!/bin/sh
# this script will take the following input:
# MODELS
# inception3 48
# resnet152 32
# ...
#
# And turn it into:
# MODELS="inception3 resnet152"
# inception3="48"
# resnet152="32"
# ...

sed -E '/#/d' $* | awk -v RS='\n\n' -v FS='\n' '
	$1 ~ /.+/ && $2 ~ /.+/ {
	for (i=2; i <= NF; i++) {
		len = split($i, arr, " ");
		if (arr[2]) {
			v=arr[2]
			for (j=3; j < len; j++) {
				v = v " " arr[j];
			}
			printf "%s=\"%s\"\n", arr[1], v
		}
		headarr[k++] = arr[1];
	}
	v=headarr[0]
	for (i=1; i < k; i++) {
		v = v " " headarr[i]
	}
	printf "%s=\"%s\"\n", $1, v
	delete headarr
	delete arr
}'
