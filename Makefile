CC=nvcc
LD=nvcc
CFLAGS=  -c -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU #-O3
LDFLAGS=   -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -lcudart #-O3
CUDAFLAGS= -G -g -c -arch=sm_21 #-O3

ALL= cudaRayTrace.o

all= $(ALL) RTRT

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o RTRT

cudaRayTrace.o:	cudaRayTrace.cu cudaRayTrace.h
	$(CC) $(CUDAFLAGS) -o $@ $< 

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

