.PHONY: all lib applications

all: lib applications

lib:
	@ mkdir -p build
	cd build && \
	cmake -DCMAKE_INSTALL_PREFIX=${FOAM_USER_LIBBIN} ../third_party/RTXAdvect && \
	make -j4 && \
	make install

applications:
	(cd applications/cudaParticlesUncoupledFoam && wmake) && \
	(cd applications/cudaParticlesPimpleFoam && wmake)

clean:
	rm -r build && \
	(cd tutorials && ./Allclean) && \
	(cd applications/cudaParticlesUncoupledFoam && wclean) && \
	(cd applications/cudaParticlesPimpleFoam && wclean)
