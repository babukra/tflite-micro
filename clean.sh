export PATH=$PATH:/opt/xtensa/XtDevTools/install/tools/RI-2022.9-linux/XtensaTools/bin
export LM_LICENSE_FILE=~/tensilica.lic
export XTENSAD_LICENSE_FILE=~/tensilica.lic
#export XTENSA_CORE=HiFi5_slate_v1_1
export XTENSA_BASE=/opt/xtensa/XtDevTools/install/tools

make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa \
	OPTIMIZED_KERNEL_DIR=xtensa \
	TARGET_ARCH=hifi5 \
	XTENSA_TOOLS_VERSION=RI-2021.7-linux \
	XTENSA_CORE=HiFi5_slate_v1_1 \
	clean
