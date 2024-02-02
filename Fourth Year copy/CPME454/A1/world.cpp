// world.cpp


#include "world.h"
#include "lander.h"
#include "ll.h"
#include "gpuProgram.h"
#include "strokefont.h"

#include <sstream>
#include <iomanip>

//const float textAspect = 0.7;	// text width-to-height ratio (you can use this for more realistic text on the screen)
int score = 0;
float timer = 0;
bool isSuccessful = false; 
bool crash = false;

void World::updateState( float elapsedTime )


{
  // See if any keys are pressed for thrust

  if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS) // right arrow
    lander->rotateCW( elapsedTime );

  if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS) // left arrow
    lander->rotateCCW( elapsedTime );

  if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS) // down arrow
    lander->addThrust( elapsedTime );

  // Update the position and velocity

  lander->updatePose( elapsedTime );

  // See if the lander has touched the terrain

  vec3 closestTerrainPoint = landscape->findClosestPoint( lander->centrePosition() );
  float closestDistance = ( closestTerrainPoint - lander->centrePosition() ).length();

  // Find if the view should be zoomed

  zoomView = (closestDistance < ZOOM_RADIUS);

  // Check for landing or collision and let the user know
  //
  // Landing is successful if the vertical speed is less than 1 m/s and
  // the horizontal speed is less than 0.5 m/s.
  //
  // SHOULD ALSO CHECK THAT LANDING SURFACE IS HORIZONAL, BUT THIS IS
  // NOT REQUIRED IN THE ASSIGNMENT.
  //
  // SHOULD ALSO CHECK THAT THE LANDER IS VERTICAL, BUT THIS IS NOT
  // REQUIRED IN THE ASSIGNMENT.

  // YOUR CODE HERE
  if (closestDistance < 5.0) {
    if (abs(lander->velocity.y) < 1.0 && abs(lander->velocity.x) < 0.5) {
      cout << "Landing successful!" << endl;
      score += 100;
      isSuccessful = true;
      pauseGame = true;

    } else {
      cout << "Crash!" << endl;
      score -= 100;
      crash = true;
      pauseGame = true;

    }
  }
  
  timer += elapsedTime;
  
}

void World::resetLander() {
  lander->reset();
  timer = 0;
  isSuccessful = false;
  crash = false;
  pauseGame = false;
}

void World::draw()

{
  mat4 worldToViewTransform;

  if (!zoomView) {

    // Find the world-to-view transform that transforms the world
    // to the [-1,1]x[-1,1] viewing coordinate system, with the
    // left edge of the landscape at the left edge of the screen, and
    // the bottom of the landscape BOTTOM_SPACE above the bottom edge
    // of the screen (BOTTOM_SPACE is in viewing coordinates).

    float s = 2.0 / (landscape->maxX() - landscape->minX());

    worldToViewTransform
      = translate( -1, -1 + BOTTOM_SPACE, 0 )
      * scale( s, s, 1 )
      * translate( -landscape->minX(), -landscape->minY(), 0 );

  } else {

    // Find the world-to-view transform that is centred on the lander
    // and is 2*ZOOM_RADIUS wide (in world coordinates).

    // YOUR CODE HERE
    float z = 1.0 / ZOOM_RADIUS;

    worldToViewTransform
      = scale( z, z, 1 ) * translate( -lander->centrePosition().x, -lander->centrePosition().y, 0 );
  }

  // Draw the landscape and lander, passing in the worldToViewTransform
  // so that they can append their own transforms before passing the
  // complete transform to the vertex shader.

  landscape->draw( worldToViewTransform );
  lander->draw( worldToViewTransform );

  // Debugging: draw line between lander and closest point

  if (showClosestPoint)
    segs->drawOneSeg( landscape->findClosestPoint( lander->centrePosition() ), 
		      lander->centrePosition(), 
		      worldToViewTransform );

  // Draw the heads-up display (i.e. all text).

  stringstream ss;

  drawStrokeString( "LUNAR LANDER", -0.2, 0.85, 0.06, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  ss.setf( ios::fixed, ios::floatfield );
  ss.precision(1);

  ss.str("");
  ss << "HORIZONTAL SPEED: " << lander->velocity.x << " m/s ->"; 
  drawStrokeString( ss.str(), -0.95, 0.75, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  ss.str("");
  ss << "VERTICAL SPEED: " <<  -lander->velocity.y << " m/s";
  drawStrokeString( ss.str(), -0.95, 0.70, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  //draw down arrow beside vertical speed
  ss.str("");
  ss << "|";
  drawStrokeString( ss.str(), -0.28, 0.71, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );
  
  ss.str("");
  ss << "v";
  drawStrokeString( ss.str(), -0.28, 0.69, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );



  // YOUR CODE HERE (modify the above code, too)
  ss.str("");
  ss << "FUEL " << lander->fuel << " kg";
  drawStrokeString( ss.str(), 0.3, 0.75, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  ss.str("");
  ss << "TIME: " << timer << " s";
  drawStrokeString( ss.str(), -0.95, 0.65, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  float altitude = -(landscape->findHeightAtX( lander->centrePosition().x ) - lander->centrePosition().y);
  ss.str("");
  ss << "ALTITUDE " << altitude  << " m";
  drawStrokeString( ss.str(), 0.3, 0.70, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  ss.str("");
  ss << "SCORE: " << score  << " POINTS";
  drawStrokeString( ss.str(), 0.3, 0.65, 0.04, glGetUniformLocation( myGPUProgram->id(), "MVP") );

  if (isSuccessful) {
    // Draw the text
    ss.str("");
    ss << "Congratulations! Press R to play again.";
    drawStrokeString(ss.str(), -0.5, 0.25, 0.04, glGetUniformLocation(myGPUProgram->id(), "MVP"));
  }else if(crash){

    // Draw the text
    ss.str("");
    ss << "You crashed! Press R to play again.";
    drawStrokeString(ss.str(), -0.5, 0.25, 0.04, glGetUniformLocation(myGPUProgram->id(), "MVP"));
    
  }

}
