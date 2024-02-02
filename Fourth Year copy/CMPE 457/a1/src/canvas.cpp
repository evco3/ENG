// canvas.cpp


#include "canvas.h"
#include "main.h"
#include "gpuProgram.h"
#include "strokefont.h"
#include "editor.h"

#include <sstream>


#define BACKGROUND_BLOCK_SIZE 20 // for background checkboard under transparent images


void Canvas::draw()

{
  // Draw image centred within window

  float x = image->width / (float) canvasWidth;
  float y = image->height / (float) canvasHeight;

  background->draw( vec2(-x,-y), vec2(x,y) );
  
  image->draw( vec2(-x,-y), vec2(x,y) );

  // Draw box around image if image is smaller than window
  //
  // Put box one pixel outside of the image

  if (image->width < canvasWidth || image->height < canvasHeight) {

    vec4 colour(0,0,0,1);
    vec3 lightDir(1,1,1);
    mat4 M = identity4();

    float pixelX = 2/(float)canvasWidth;
    float pixelY = 2/(float)canvasHeight;
    
    vec3 pts[4] = { vec3( -x-pixelX, -y-pixelY, 0 ), 
		    vec3(  x+pixelX, -y-pixelY, 0 ),
		    vec3(  x+pixelX,  y+pixelY, 0 ),
		    vec3( -x-pixelX,  y+pixelY, 0 ) };
    
    segs->drawSegs( GL_LINE_LOOP, pts, colour, NULL, 4, M, M, lightDir );
  }

  // Draw status message

  string modeStr;
  
  if (editor->editMode == TRANSLATE)
    modeStr = "translate";
  else if (editor->editMode == ROTATE)
    modeStr = "rotate";
  else if (editor->editMode == SCALE)
    modeStr = "scale";
  else if (editor->editMode == INTENSITY)
    modeStr = "intensity";
  else
    modeStr = "";

  strokeFont->drawStrokeString( modeStr.c_str(), -0.98, -0.95, 0.06, 0, LEFT );

  string projStr;
  
  if (editor->projectionMode == FORWARD)
    projStr = "forward";
  else if (editor->projectionMode == BACKWARD)
    projStr = "backward";
  else
    projStr = "";

  strokeFont->drawStrokeString( projStr.c_str(), 0.98, -0.95, 0.06, 0, RIGHT );
}


// Create a checkerboard texture to be used as background below
// transparent images

Texture * Canvas::setupBackgroundTexture( unsigned int width, unsigned int height ) 

{
  Texture *tex = new Texture( width, height );

  for (unsigned int x=0; x<width; x++)
    for (unsigned int y=0; y<height; y++) {

      Pixel &p = tex->pixel(x,y);
	
      if ((x/BACKGROUND_BLOCK_SIZE + y/BACKGROUND_BLOCK_SIZE) % 2 == 0)
	p.r = p.g = p.b = 230; // grey
      else
	p.r = p.g = p.b = 255; // white

      p.a = 255; // opaque
    }

  return tex;
}


// Shaders for canvas rendering


char *Canvas::vertexShader = R"XX(

#version 300 es

uniform mediump mat4 MVP;

layout (location = 0) in mediump vec4 position;
layout (location = 1) in mediump vec4 colour_in;

out mediump vec4 colour;


void main()

{
  gl_Position = MVP * position;
  colour = colour_in;
}

)XX";



char *Canvas::fragmentShader = R"XX(

#version 300 es

in mediump vec4 colour;

out mediump vec4 fragColour;


void main()

{
  fragColour = colour;
}

)XX";



