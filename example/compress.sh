#!/bin/bash
cd "$(dirname $0)"

# make clean
rm -rf data.lqa/

# data.vti is a 128**3 voxel data set, with no periodic boundarys
# contains two variables: vel vector, obs scalar

# this script does not require liteqa to be installed
# specify location of liteqa_c shared object, if not installed
[ ! -e ../liteqa_c/liteqa_c.so ] && ( cd ../liteqa_c/ ; make )
export LITEQA_C=../liteqa_c/liteqa_c.so

# liteqa auto-generates two files liteqa-hilbmap.64 and liteqa-hilbmap.128
# those are hilbert coordinates, which will be generated once and
# reloaded, make sure they are located in the current working
# directory, otherwise liteqa will regenerate them

function run() {
	DATA=$1

	## GENERATE COMPRESSED INDEX

	# generate compressed index for flow field x component
	../vti2liteqa.py --index --icube 128 --vtifile $DATA.vti:vel --component 0 --maskfile $DATA.vti:obs --lqadir $DATA.lqa/ --lqaarray vel --check

	# generate compressed index for velocity magnitude
	../vti2liteqa.py --index --icube 128 --vtifile $DATA.vti:vel --trafo mag --maskfile $DATA.vti:obs --lqadir $DATA.lqa/ --lqaarray vmag --check

	# generate compressed index for velocity q-criterion
	../vti2liteqa.py --index --icube 128 --vtifile $DATA.vti:vel --trafo qcrit --maskfile $DATA.vti:obs --lqadir $DATA.lqa/ --lqaarray qcrit --check

	# generate compressed index for velocity vorticity
	../vti2liteqa.py --index --icube 128 --vtifile $DATA.vti:vel --trafo vort+mag --maskfile $DATA.vti:obs --lqadir $DATA.lqa/ --lqaarray vort --check

	# generate compressed index for mask distance field
	../vti2liteqa.py --index --icube 128 --vtifile $DATA.vti:obs --trafo dist --maskfile $DATA.vti:obs --lqadir $DATA.lqa/ --lqaarray dist --check

	## GENERATE COMPRESSED GRID

	# generate compressed grid for flow field vectors
	../vti2liteqa.py --grid --gcube 32 --vtifile $DATA.vti:vel --component 0 --lqadir $DATA.lqa/ --lqaarray vel --check
	../vti2liteqa.py --grid --gcube 32 --vtifile $DATA.vti:vel --component 1 --lqadir $DATA.lqa/ --lqaarray vel --check
	../vti2liteqa.py --grid --gcube 32 --vtifile $DATA.vti:vel --component 2 --lqadir $DATA.lqa/ --lqaarray vel --check

	# generate compressed grid for obstacle mask
	../vti2liteqa.py --zstd --gcube 32 --vtifile $DATA.vti:obs --lqadir $DATA.lqa/ --lqaarray obs --check

	## COMPRESSION RATE
	echo "COMPRESSION RATE"

	for i in $DATA.lqa/vel0.i \
		$DATA.lqa/vel0.i \
		$DATA.lqa/dist.i \
		$DATA.lqa/qcrit.i \
		$DATA.lqa/vmag.i \
		$DATA.lqa/vort.i \
		$DATA.lqa/vel0.g \
		$DATA.lqa/vel1.g \
		$DATA.lqa/vel2.g \
	#
	do
		python -c "import os; print('$i', '%.2f %%' % (os.path.getsize('$i')/(128**3*4)*100))"
	done

	python -c "import os; print('$DATA.lqa/obs.g', '%.2f %%' % (os.path.getsize('$DATA.lqa/obs.g')/(128**3)*100))"
}

## RUN COMPRESSION

run data1
run data2
