//callbacksPBO.cpp (Rob Farber)

#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>

// variables for keyboard control
int animFlag=1;
float animTime=0.0f;
float animInc=0.1f;

// variables for mouse control
//int lastx = -1;
//int lasty = -1;

//external variables
extern GLuint pbo;
extern GLuint textureID;
extern unsigned int image_width;
extern unsigned int image_height;
extern void moveIn();
extern void moveOut();
extern void moveUp();
extern void moveDown();
extern void moveLeft();
extern void moveRight();

extern "C" void wasdMove(unsigned char key);

extern "C" void ijklMove(int x, int y);
extern "C" void misc(unsigned char key);
// The user must create the following routines:
void runCuda();

void display()
{
   // run CUDA kernel
   runCuda();

   // Create a texture from the buffer
   glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);

   // bind texture from PBO
   glBindTexture(GL_TEXTURE_2D, textureID);


   // Note: glTexSubImage2D will perform a format conversion if the
   // buffer is a different format from the texture. We created the
   // texture with format GL_RGBA8. In glTexSubImage2D we specified
   // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

   // Note: NULL indicates the data resides in device memory
   glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
         GL_RGBA, GL_UNSIGNED_BYTE, NULL);


   // Draw a single Quad with texture coordinates for each vertex.

   glBegin(GL_QUADS);
   glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,0.0f,0.0f);
   glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,1.0f,0.0f);
   glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,0.0f,0.0f);
   glEnd();

   // Don't forget to swap the buffers!
   glutSwapBuffers();

   // if animFlag is true, then indicate the display needs to be redrawn
   if(animFlag) {
      glutPostRedisplay();
      animTime += animInc;
   }
}

//! Keyboard events handler for GLUT
void keyboard(unsigned char key, int x, int y)
{
   switch(key) {
   case(27) :
      exit(0);
      break;
   case('w'):
   case('a'):
   case('s'):
   case('d'):
     wasdMove(key);
     break;
   /*case('i'):
   case('k'):
   case('j'):
   case('l'):
     ijklMove(key);*/
     break;
   case('q'):
   case('r'):
   case('-'):
   case('='):
   case('o'):
   case('9'):
   case('0'):
   case('p'):
   case('['):
   case(']'):
     misc(key);
     break;
   }
   ijklMove(1, 2);
// indicate the display must be redrawn
   glutPostRedisplay();
}
void mouse(int button, int state, int x, int y)
{


}
void motion(int x, int y)
{

   ijklMove(x, y);
//   printf("Well at least I can do something\n");

}
