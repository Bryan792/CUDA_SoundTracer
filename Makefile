CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU 
LDFLAGS= -O3  -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -lcudart
CUDAFLAGS= -O3 -c -arch=sm_21

ALL= callbacksPBO.o cudaRayTrace.o simpleGLmain.o simplePBO.o

all= $(ALL) RTRT

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o RTRT

callbacksPBO.o:	callbacksPBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

kernelPBO.o:	kernelPBO.cu
	$(CC) $(CUDAFLAGS) -o $@ $<

cudaRayTrace.o:	cudaRayTrace.cu cudaRayTrace.h
	$(CC) $(CUDAFLAGS) -o $@ $< fmod/lib/libfmodex64.so 


simpleGLmain.o:	simpleGLmain.cpp
	$(CC) $(CFLAGS) -o $@ $<

simplePBO.o: simplePBO.cpp
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

