#!/bin/bash

DIRECTORY="dataset"
PARALLEL_JOBS=4

if [ -d "${DIRECTORY}" ]; then
	echo "Recursively converting all files in ${DIRECTORY} to .jpg"
	find "${DIRECTORY}" -type f -print0 | xargs -0 -P ${PARALLEL_JOBS} mogrify -format jpg

	echo "Deleting files that do not have the .jpg extension"
	find "${DIRECTORY}" -type f ! -name "*.jpg" -print0 | xargs -0 -P ${PARALLEL_JOBS} rm
else
	echo "${DIRECTORY} does not exist. Exiting."
fi
