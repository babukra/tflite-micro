#export VERSION=2021.8
export VERSION=2021.7
export BENCHMARK=run_person_detection_benchmark
#export BENCHMARK=run_wav2letter_benchmark
#export BENCHMARK=run_keyword_benchmark
#export BENCHMARK=run_lstm_benchmark
#export XTENSA_CORE=nxp_rt600_RI2021_8_xclib
export XTENSA_CORE=HiFi5_slate_v1_1
#export CORE=hifi4
export CORE=hifi5

export PATH=$PATH:/opt/xtensa/XtDevTools/install/tools/RI-$VERSION-linux/XtensaTools/bin
export LM_LICENSE_FILE=~/tensilica.lic
export XTENSAD_LICENSE_FILE=~/tensilica.lic
export XTENSA_BASE=/opt/xtensa/XtDevTools/install/tools
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa \
	OPTIMIZED_KERNEL_DIR=xtensa \
	TARGET_ARCH=$CORE \
	XTENSA_TOOLS_VERSION=RI-$VERSION-linux \
	XTENSA_CORE=$XTENSA_CORE \
	$BENCHMARK -j18
