// base
#include "particle.h"
//#include <vector>  // The vector header is already included in particle.h

Particle::Particle(size_t numParticles)
 : mMass(numParticles, 0.0f),
   mDensity(numParticles, 0.0f),
   mPosition(numParticles * 3, 0.0f),
   mVelocity(numParticles * 3, 0.0f),
   mAcceleration(numParticles * 3, 0.0f),
   mNeighborCount(numParticles, 0)
   //Maybe internalEnergy? -> Para el cooling (necesito tabla de rate(rho, u_int) (!))
{
   
}