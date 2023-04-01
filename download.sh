make -f tensorflow/lite/micro/tools/make/Makefile TARGET=xtensa \
	XTENSA_TOOLS_VERSION=RI-2022.9-linux \
	XTENSA_BASE=/opt/xtensa/XtDevTools/install/tools \
	XTENSA_CORE=hifi5_ao_7 \
	TARGET_ARCH=hifi5 \
	OPTIMIZED_KERNEL_DIR=xtensa \
	third_party_downloads
