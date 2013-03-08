CC=nvcc
LD=nvcc
CFLAGS= -O3 -c -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU 
LDFLAGS= -O3  -lGL -lglut -DGL_GLEXT_PROTOTYPES -lGLU -lcudart
CUDAFLAGS= -O3 -c -arch=sm_21

ALL= cudaRayTrace.o

all= $(ALL) RTRT

RT:	$(ALL)
	$(CC) $(LDFLAGS) $(ALL) -o RTRT

cudaRayTrace.o:	cudaRayTrace.cu cudaRayTrace.h
	$(CC) $(CUDAFLAGS) -o $@ $< 

clean:
	rm -rf core* *.o *.gch $(ALL) junk*

