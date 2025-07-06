// base
#include "sph.h"

// sph
#include "particle.h"

// Qt
#include <QElapsedTimer>

// cmath
#include <math.h>

// openmp
#include <omp.h>

#include <QDateTime>

// write
#include <iostream>
#include <fstream>
#include <sys/stat.h> 
#include <sys/types.h> // write
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include <immintrin.h>

#ifndef M
#define M 64
#endif
#define K 8

// Unidades: [km/s pc M_sun Myr]...
// ¿Define a cada step estos valores?
SPH::SPH()
 : mParticleCount(0),
   mGridCellCount(0),
   mRho0(0.0f),
   mStopped(false),
   mPaused(false)
{
   // Grid
   mH = 0.1f;  // OG = 3.34;
   mH2 = pow(mH, 2);
   mH6 = pow(mH, 6);
   mH9 = pow(mH, 9);
   mHTimes2 = mH * 2.0f;
   mHTimes2Inv = 1.0f / mHTimes2;
   // ¿Pero no conviene definirlos como constantes?
   mParticleCount = M * 1024;
   mGridCellsX = 24;  // OG 32...
   mGridCellsY = 24;
   mGridCellsZ = 24;
   mGridCellCount = mGridCellsX * mGridCellsY * mGridCellsZ;
   mCellSize = 2.0f * mH;
   mMaxX = mCellSize * mGridCellsX;
   mMaxY = mCellSize * mGridCellsY;
   mMaxZ = mCellSize * mGridCellsZ;

   float time_simu = 3.0f;  // [Myr]
   mTimeStep = 1e-4f;
   totalSteps = (int)round(time_simu/mTimeStep);

   // Physics
   mGravity = vec3(0.0f, 0.0f, 0.0f);  // Legacy. Only if we want e.g. a "wind" or Earth condition.
   mViscosityScalar = 1e+1f;
   mDamping = 1.0f;  // Deberíamos "tirar" las que se escapen (En vez de checkear boundaries...)
   // Grav (and/or central pot)
   mGravConstant = 4.3009e-3f;  // En pc (km/s)^2 / M_sun
   mCentralMass = 1e+5f;  // As we wish
   mCentralPos[0] = mMaxX * 0.5f;
   mCentralPos[1] = mMaxY * 0.5f;
   mCentralPos[2] = mMaxZ * 0.5f;
   mSoftening = mH; // * 0.5;  // Ojo, se recomienda que sea = h...

   // mass = 1 M_sun; Si hay más, menos masa c/u?
   float mass = 1.0f;// * (64/M);  // -> No normalizo porque idem vecinos pero menos masivos => menos visocidad...
   mCflLimit = 10000.0f;  // Esto no debería existir...
   mCflLimit2 = mCflLimit * mCflLimit;

   // NEW: EoS
   mA_fluid = 1e-1f;  // Check S&H03
   mGamma_minus = 5.f/3.f - 1.f;
   mInvGamma_minus = 1.f/mGamma_minus;
   // Or more stiff:
   mRho0 = 1e+1f;
   mStiffness = 1e-1f;

   // Feedback threshold? (& kick)
   mRho_thresh = 1e+4f;
   mKickVel = 100.0f;  // Try 1

   // smoothing kernels
   mKernel1Scaled = 315.0f / (64.0f * M_PIf * mH9);
   mKernel2Scaled = -45.0f / (M_PIf * mH6);  // M_PI as float?
   mKernel3Scaled = -mKernel2Scaled;

   // Valor fiducial = 32
   mExamineCount = 32;

   mSrcParticles = new Particle(mParticleCount);
   mVoxelIds= new int[mParticleCount];
   mVoxelCoords= new vec3i[mParticleCount];

   // Para difs masas (estaria bueno ver que onda...)
   for (int i = 0; i < mParticleCount; i++)
   {
      mSrcParticles->mMass[i] = mass;
   }

   mGrid = new QList<uint32_t>[mGridCellCount];

   mNeighbors = new uint32_t[mParticleCount*mExamineCount];
   mNeighborDistancesScaled = new float[mParticleCount*mExamineCount];

   // Randomize particle start positions (rotating)
   //initRotatingGasCloud();

   // Two non-centered spheres (orbiting a centre)
   //initTwoSpheres();

   // Something with a bounding box? e.g. splash of water
   initBlob();

   // Maybe add here una opcion para cargar las condiciones de un archivo externo?
}

SPH::~SPH()
{
   stopSimulation();
   quit();
   wait();
}


bool SPH::isStopped() const
{
   mMutex.lock();
   bool stopped = mStopped;
   mMutex.unlock();

   return stopped;
}


bool SPH::isPaused() const
{
   mMutex.lock();
   bool paused = mPaused;
   mMutex.unlock();

   return paused;
}



void SPH::run()
{
   int stepCount = 0;

   // Create directory ./out
   /*
   const char *path = "out";
   int result = mkdir(path, 0777);
   if (result == 0)
      std::cout << "Directory created" << std::endl;
   else
      std::cout << "Directory already exists" << std::endl;

   // Create files
   std::ofstream outfile1("out/energy.txt");
   outfile1 << "Step, Kinetic Energy, Potential Energy, Total Energy" << std::endl;
   std::ofstream outfile2("out/angularmomentum.txt");
   outfile2 << "Step, Angular Momentum" << std::endl;
   std::ofstream outfile3("out/timing.txt");
   outfile3 << "Step, Voxelize, Find Neighbors, Compute Density, Compute Pressure, Compute Acceleration, Integrate" << std::endl;
   std::ofstream outfile4("out/neighbors.txt");
   */


   while(!isStopped() && stepCount <= totalSteps)
   {
      if (!isPaused())
      {
         step(stepCount);
         /*
         outfile1 << stepCount << ", " << mKineticEnergyTotal << ", " << mPotentialEnergyTotal << ", " << mKineticEnergyTotal + mPotentialEnergyTotal << std::endl;
         outfile2 << stepCount << ", " << mAngularMomentumTotal.length() << std::endl;
         outfile3 << stepCount << ", " << timeVoxelize << ", " << timeFindNeighbors << ", " << timeComputeDensity << ", " << timeComputePressure << ", " << timeComputeAcceleration << ", " << timeIntegrate << std::endl;
         */
         stepCount++;
      }
   }

   /*
   outfile1.close();
   outfile2.close();
   outfile3.close();
   outfile4.close();
   */
}


void SPH::step(int this_step)
{
   timeVoxelize = 0;
   timeFindNeighbors = 0;
   timeComputeDensity = 0;
   timeComputePressure = 0;
   timeComputeAcceleration = 0;
   timeIntegrate = 0;
   QElapsedTimer t;

   // put particles into voxel grid
   voxelizeParticles();

   #pragma omp parallel 
   {
      // find neighboring particles
      #pragma omp for schedule(guided)
      for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
      {
         const vec3i& voxel= mVoxelCoords[particleIndex];

         // neighbors for this particle
         uint32_t* neighbors= &mNeighbors[particleIndex*mExamineCount];
         // Calc 2 times dist a neighbors? Let's do it here:
         float* neighborDistances= &mNeighborDistancesScaled[particleIndex*mExamineCount];

         findNeighbors(particleIndex, neighbors, voxel.x, voxel.y, voxel.z, neighborDistances);

         computeDensity(particleIndex, neighbors, neighborDistances);
      }

      // compute acceleration
      #pragma omp for schedule(guided)
      for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
      {
         // neighbors for this particle
         uint32_t* neighbors= &mNeighbors[particleIndex*mExamineCount];
         float* neighborDistances= &mNeighborDistancesScaled[particleIndex*mExamineCount];

         computeAcceleration(particleIndex, neighbors, neighborDistances);

         // Puedo poner acá el kick de feedback <=> Quiero feedback...
         /*
         if (this_step % 10 == 0)
         {
            computeFeedback(particleIndex, neighbors, neighborDistances);
         }
         */
      }

      // integrate (after computeAccel for the system)
      #pragma omp for schedule(guided)
      for (int particleIndex = 0; particleIndex < mParticleCount; particleIndex++)
      {
         // Debería la integración hacerce cargo de aplicar los boundaries?
         integrate(particleIndex);
      }
   }

   emit updateElapsed(
      timeVoxelize,
      timeFindNeighbors,
      timeComputeDensity,
      timeComputePressure,
      timeComputeAcceleration,
      timeIntegrate
   );

   emit stepFinished();
}


void SPH::pauseResume()
{
   mMutex.lock();
   mPaused = !mPaused;
   mMutex.unlock();
}


void SPH::stopSimulation()
{
   mMutex.lock();
   mStopped = true;
   mMutex.unlock();
}


// Sobra... -> Cambiar por otras cond inic
void SPH::initTwoSpheres()
{
   // Fix seed:
   srand(42);

   // 1st sphere
   float sphereCenter_x_1 = mMaxX * 0.05f;
   float sphereCenter_y_1 = mMaxY * 0.05f;
   float sphereCenter_z_1 = mMaxZ * 0.05f;
   // 2nd sphere
   float sphereCenter_x_2 = mMaxX * 0.95f;
   float sphereCenter_y_2 = mMaxY * 0.95f;
   float sphereCenter_z_2 = mMaxZ * 0.95f;

   // Vars
   float dist = 0.0f;
   float x = 0.0f;
   float y = 0.0f;
   float z = 0.0f;

   float radius = 1.0f;  // Each sphere
   float phi;  // El ang acimutal para la v_tangencial. (atan2(y,x))
   float v_x_inic, v_y_inic, v_z_inic;  // El hdp puso a y como la comp vertical...
                              // (no quiero v_inic en "z" (que aca es "y"))

   // 1st sphere
   for (int i = 0; i < mParticleCount/2; i++)
   {
      do
      {
         x = rand() / (float)RAND_MAX;
         y = rand() / (float)RAND_MAX;
         z = rand() / (float)RAND_MAX;

         x *= mGridCellsX * mHTimes2;
         y *= mGridCellsY * mHTimes2;
         z *= mGridCellsZ * mHTimes2;

         if (x == (float)mGridCellsX)
            x -= 0.00001f;
         if (y == (float)mGridCellsY)
            y -= 0.00001f;
         if (z == (float)mGridCellsZ)
            z -= 0.00001f;

         //dist = (vec3(x,y,z) - sphereCenter).length();
         dist = (x - sphereCenter_x_1) * (x - sphereCenter_x_1) +\
                (y - sphereCenter_y_1) * (y - sphereCenter_y_1) +\
                (z - sphereCenter_z_1) * (z - sphereCenter_z_1);
         dist = sqrtf(dist);
      }
      while (dist > radius);

      mSrcParticles->mPosition[i * 3] = x;
      mSrcParticles->mPosition[i * 3 + 1] = y;
      mSrcParticles->mPosition[i * 3 + 2] = z;

      // Orbital vel parameter
      phi = atan2(z - mMaxZ * 0.5f, x - mMaxX * 0.5f);  // Acomodar por el centro del box
      dist = (sphereCenter_x_1 - mMaxZ * 0.5f) * (sphereCenter_x_1 - mMaxZ * 0.5f) +\
            (sphereCenter_y_1 - mMaxZ * 0.5f) * (sphereCenter_y_1 - mMaxZ * 0.5f) +\
            (sphereCenter_z_1 - mMaxZ * 0.5f) * (sphereCenter_z_1 - mMaxZ * 0.5f);
      v_x_inic = 2.0f * sqrtf(dist) * -sin(phi);  // a = 20.0
      v_z_inic = 2.0f * sqrtf(dist) * cos(phi);  // a = 20.0

      // Random vels =/= rotating, but orbital velocity
      v_x_inic += ((rand() / (float)RAND_MAX) * 1.f) - 0.5f;
      v_y_inic = ((rand() / (float)RAND_MAX) * 1.f) - 0.5f;
      v_z_inic += ((rand() / (float)RAND_MAX) * 1.f) - 0.5f;
      // Power law w.r.t. to distance to the sphere's centre
      //v_z_inic = 20.0f * pow(dist + mH*0.5, -0.5) * cos(phi);  // a = 20.0

      //mSrcParticles->mVelocity[i].set(v_x_inic, v_y_inic, v_z_inic);
      mSrcParticles->mVelocity[i * 3] = v_x_inic;
      mSrcParticles->mVelocity[i * 3 + 1] = v_y_inic;
      mSrcParticles->mVelocity[i * 3 + 2] = v_z_inic;
   }

   // 2nd sphere
   for (int i = mParticleCount/2; i < mParticleCount; i++)
   {
      do
      {
         x = rand() / (float)RAND_MAX;
         y = rand() / (float)RAND_MAX;
         z = rand() / (float)RAND_MAX;

         x *= mGridCellsX * mHTimes2;
         y *= mGridCellsY * mHTimes2;
         z *= mGridCellsZ * mHTimes2;

         if (x == (float)mGridCellsX)
            x -= 0.00001f;
         if (y == (float)mGridCellsY)
            y -= 0.00001f;
         if (z == (float)mGridCellsZ)
            z -= 0.00001f;

         //dist = (vec3(x,y,z) - sphereCenter).length();
         dist = (x - sphereCenter_x_2) * (x - sphereCenter_x_2) +\
                (y - sphereCenter_y_2) * (y - sphereCenter_y_2) +\
                (z - sphereCenter_z_2) * (z - sphereCenter_z_2);
         dist = sqrtf(dist);
      }
      while (dist > radius);

      mSrcParticles->mPosition[i * 3] = x;
      mSrcParticles->mPosition[i * 3 + 1] = y;
      mSrcParticles->mPosition[i * 3 + 2] = z;

      // Orbital vel parameter
      phi = atan2(z - mMaxZ * 0.5f, x - mMaxX * 0.5f);  // Acomodar por el centro del box
      dist = (sphereCenter_x_2 - mMaxZ * 0.5f) * (sphereCenter_x_2 - mMaxZ * 0.5f) +\
            (sphereCenter_y_2 - mMaxZ * 0.5f) * (sphereCenter_y_2 - mMaxZ * 0.5f) +\
            (sphereCenter_z_2 - mMaxZ * 0.5f) * (sphereCenter_z_2 - mMaxZ * 0.5f);
      v_x_inic = 2.0f * sqrtf(dist) * -sin(phi);  // a = 20.0
      v_z_inic = 2.0f * sqrtf(dist) * cos(phi);  // a = 20.0

      // Random vels =/= rotating, but orbital velocity
      v_x_inic += ((rand() / (float)RAND_MAX) * 1.f) - 0.5f;
      v_y_inic = ((rand() / (float)RAND_MAX) * 1.f) - 0.5f;
      v_z_inic += ((rand() / (float)RAND_MAX) * 1.f) - 0.5f;

      // Power law w.r.t. to distance to the sphere's centre
      //v_z_inic = 20.0f * pow(dist + mH*0.5, -0.5) * cos(phi);  // a = 20.0

      //mSrcParticles->mVelocity[i].set(v_x_inic, v_y_inic, v_z_inic);
      mSrcParticles->mVelocity[i * 3] = v_x_inic;
      mSrcParticles->mVelocity[i * 3 + 1] = v_y_inic;
      mSrcParticles->mVelocity[i * 3 + 2] = v_z_inic;

      //mSrcParticles->mVelocity[i].set(v_x_inic, v_y_inic, v_z_inic);
      mSrcParticles->mVelocity[i * 3] = v_x_inic;
      mSrcParticles->mVelocity[i * 3 + 1] = v_y_inic;
      mSrcParticles->mVelocity[i * 3 + 2] = v_z_inic;
   }
}


void SPH::initRotatingGasCloud()
{
   // Fix seed:
   srand(42);

   float dist = 0.0f;

   float x = 0.0f;
   float y = 0.0f;
   float z = 0.0f;

   float sphereCenter_x = mMaxX * 0.5f;
   float sphereCenter_y = mMaxY * 0.5f;
   float sphereCenter_z = mMaxZ * 0.5f;

   float radius = 2.0f;
   float phi;  // El ang acimutal para la v_tangencial. (atan2(y,x))
   float v_x_inic, v_y_inic, v_z_inic;  // El hdp puso a y como la comp vertical...
                              // (no quiero v_inic en "z" (que aca es "y"))

   for (int i = 0; i < mParticleCount; i++)
   {
      do
      {
         x = rand() / (float)RAND_MAX;
         y = rand() / (float)RAND_MAX;
         z = rand() / (float)RAND_MAX;

         x *= mGridCellsX * mHTimes2;
         y *= mGridCellsY * mHTimes2;
         z *= mGridCellsZ * mHTimes2;

         if (x == (float)mGridCellsX)
            x -= 0.00001f;
         if (y == (float)mGridCellsY)
            y -= 0.00001f;
         if (z == (float)mGridCellsZ)
            z -= 0.00001f;

         //dist = (vec3(x,y,z) - sphereCenter).length();
         dist = (x - sphereCenter_x) * (x - sphereCenter_x) +\
                (y - sphereCenter_y) * (y - sphereCenter_y) +\
                (z - sphereCenter_z) * (z - sphereCenter_z);
         dist = sqrtf(dist);
      }
      while (dist > radius);

      mSrcParticles->mPosition[i * 3] = x;
      mSrcParticles->mPosition[i * 3 + 1] = y;
      mSrcParticles->mPosition[i * 3 + 2] = z;

      phi = atan2(z - mMaxZ * 0.5f, x - mMaxX * 0.5f);  // Acomodar por el centro de la esfera!
      v_x_inic = 20.0f * pow(dist + mH*0.5, -0.5) * -sin(phi);  // a = 20.0
      v_z_inic = 20.0f * pow(dist + mH*0.5, -0.5) * cos(phi);  // a = 20.0

      // Some random movements on "y" (z)
      /*
      if (v_x_inic > 0)
      {
         v_y_inic = dist * (((rand() / (float)RAND_MAX) * 2.0f) - 0.0f);
      }
      else
      {
         v_y_inic = -dist * (((rand() / (float)RAND_MAX) * 2.0f) - 0.0f);
      }
      */

      // Change it a little (V2):
      if (phi > 0 && phi < 3.14) {
         v_y_inic = ((rand() / (float)RAND_MAX) * 6.5f);
      }
      else {
         v_y_inic = -((rand() / (float)RAND_MAX) * 6.5f);
      };

      //mSrcParticles->mVelocity[i].set(v_x_inic, v_y_inic, v_z_inic);
      mSrcParticles->mVelocity[i * 3] = v_x_inic;
      mSrcParticles->mVelocity[i * 3 + 1] = v_y_inic;
      mSrcParticles->mVelocity[i * 3 + 2] = v_z_inic;
   }

}


// Another one, a blob of a fluid with initial velocity, enclosed in a box
void SPH::initBlob()
{
   // Fix seed:
   srand(42);

   float dist = 0.0f;

   float x = 0.0f;
   float y = 0.0f;
   float z = 0.0f;

   float sphereCenter_x = mCellSize; //mMaxX * 0.5f;
   float sphereCenter_y = mMaxY * 0.5f;
   float sphereCenter_z = mMaxZ * 0.5f;

   float radius = mCellSize * 3.f;

   float v_x_inic, v_y_inic, v_z_inic;

   for (int i = 0; i < mParticleCount; i++)
   {
      do
      {
         x = rand() / (float)RAND_MAX;
         y = rand() / (float)RAND_MAX;
         z = rand() / (float)RAND_MAX;

         x *= mCellSize; //mGridCellsX * mHTimes2;
         y *= mGridCellsY * mHTimes2;
         z *= mGridCellsZ * mHTimes2;

         if (x == (float)mGridCellsX)
            x -= 0.00001f;
         if (y == (float)mGridCellsY)
            y -= 0.00001f;
         if (z == (float)mGridCellsZ)
            z -= 0.00001f;

         //dist = (vec3(x,y,z) - sphereCenter).length();
         dist = (x - sphereCenter_x) * (x - sphereCenter_x) +\
                (y - sphereCenter_y) * (y - sphereCenter_y) +\
                (z - sphereCenter_z) * (z - sphereCenter_z);
         dist = sqrtf(dist);
      }
      while (dist > radius);

      mSrcParticles->mPosition[i * 3] = x;
      mSrcParticles->mPosition[i * 3 + 1] = y;
      mSrcParticles->mPosition[i * 3 + 2] = z;

      v_x_inic = 50/sqrtf(y * y + z * z + mSoftening);
      // Random movement in the other directions
      float factor_vel = 5.f;
      v_y_inic = ((rand() / (float)RAND_MAX) * factor_vel) - factor_vel*0.5f;
      v_z_inic = ((rand() / (float)RAND_MAX) * factor_vel) - factor_vel*0.5f;

      //mSrcParticles->mVelocity[i].set(v_x_inic, v_y_inic, v_z_inic);
      mSrcParticles->mVelocity[i * 3] = v_x_inic;
      mSrcParticles->mVelocity[i * 3 + 1] = v_y_inic;
      mSrcParticles->mVelocity[i * 3 + 2] = v_z_inic;
   }

}



void SPH::clearGrid()
{
   for (int i = 0; i < mGridCellCount; i++)
   {
      mGrid[i].clear();
   }
}


void SPH::voxelizeParticles()
{
   clearGrid();

   #pragma omp parallel for
   for (int i = 0; i < mParticleCount; i++)
   {
      // compute a scalar voxel id from a position
      //vec3 pos = mSrcParticles->mPosition[i];
      float pos[3];
      pos[0] = mSrcParticles->mPosition[i * 3];
      pos[1] = mSrcParticles->mPosition[i * 3 + 1];
      pos[2] = mSrcParticles->mPosition[i * 3 + 2];

      int voxelX = (int)floor(pos[0] * mHTimes2Inv);
      int voxelY = (int)floor(pos[1] * mHTimes2Inv);
      int voxelZ = (int)floor(pos[2] * mHTimes2Inv);

      // it has been seen the positions can run slightly out of bounds for
      // one solver step. so the positions are temporarily fixed here.
      if (voxelX < 0) voxelX= 0;
      if (voxelY < 0) voxelY= 0;
      if (voxelZ < 0) voxelZ= 0;
      if (voxelX >= mGridCellsX) voxelX= mGridCellsX-1;
      if (voxelY >= mGridCellsY) voxelY= mGridCellsY-1;
      if (voxelZ >= mGridCellsZ) voxelZ= mGridCellsZ-1;

      // don't write into particle but into separate memory
      mVoxelCoords[i].x= voxelX;
      mVoxelCoords[i].y= voxelY;
      mVoxelCoords[i].z= voxelZ;

      int voxelId = computeVoxelId(voxelX, voxelY, voxelZ);

      mVoxelIds[i]= voxelId;
   }

   // put each particle into according voxel (sequential)
   for (int i = 0; i < mParticleCount; i++)
   {
       //holaxd
       mGrid[ mVoxelIds[i] ].push_back(i);
   }
}


void SPH::findNeighbors(int particleIndex, uint32_t* neighbors, int voxelX, int voxelY, int voxelZ, float* neighborDistances)
{
   float xOrientation = 0.0f;
   float yOrientation = 0.0f;
   float zOrientation = 0.0f;

   int x = 0;
   int y = 0;
   int z = 0;

   int neighborIndex = 0;
   bool enoughNeighborsFound = false;

   //vec3 pos = mSrcParticles->mPosition[particleIndex];
   float pos[3];
   pos[0] = mSrcParticles->mPosition[particleIndex * 3];
   pos[1] = mSrcParticles->mPosition[particleIndex * 3 + 1];
   pos[2] = mSrcParticles->mPosition[particleIndex * 3 + 2];

   // this gives us the relative position; i.e the orientation within a voxel
   xOrientation = pos[0] - (voxelX * mHTimes2);
   yOrientation = pos[1] - (voxelY * mHTimes2);
   zOrientation = pos[2] - (voxelZ * mHTimes2);

   // get neighbour voxels
   x = 0;
   y = 0;
   z = 0;

   (xOrientation > mH) ? x++ : x--;
   (yOrientation > mH) ? y++ : y--;
   (zOrientation > mH) ? z++ : z--;

   // neighbour voxels
   int vx[8];
   int vy[8];
   int vz[8];

   // same slice
   vx[0] = voxelX;
   vy[0] = voxelY;
   vz[0] = voxelZ;

   // distance 1
   vx[1] = voxelX + x;
   vy[1] = voxelY;
   vz[1] = voxelZ;

   vx[2] = voxelX;
   vy[2] = voxelY + y;
   vz[2] = voxelZ;

   vx[3] = voxelX;
   vy[3] = voxelY;
   vz[3] = voxelZ + z;

   // distance 2
   vx[4] = voxelX + x;
   vy[4] = voxelY + y;
   vz[4] = voxelZ;

   vx[5] = voxelX + x;
   vy[5] = voxelY;
   vz[5] = voxelZ + z;

   vx[6] = voxelX;
   vy[6] = voxelY + y;
   vz[6] = voxelZ + z;

   // distance 3
   vx[7] = voxelX + x;
   vy[7] = voxelY + y;
   vz[7] = voxelZ + z;

   int vxi;
   int vyi;
   int vzi;

   // Para ahorrarse tirar randoms, entre 0 y 4~5 -> LCG?
   int linear_cong_gen;
   int almost_a_random = 0;
   // Los def acá?
   float pos_neighbor[3];
   float dot;
   //float distanceScaled;

   // Variables para usar dentro del loop
   __m256i zeros = _mm256_setzero_si256();
   __m256i ka = _mm256_set_epi32(8,8,8,8,8,8,8,8);
   __m256i jota = _mm256_set_epi32(7,6,5,4,3,2,1,0);  //valor para los 8 jota's
   __m256i ones = _mm256_set1_epi32(1);
   __m256 mH2vec = _mm256_set1_ps(mH2);

   for (int voxelIndex = 0; voxelIndex < 8; voxelIndex++)
   {
      
      vxi = vx[voxelIndex];
      vyi = vy[voxelIndex];
      vzi = vz[voxelIndex];

      // check if voxels can be processed
      if (
            vxi > 0 && vxi < mGridCellsX
         && vyi > 0 && vyi < mGridCellsY
         && vzi > 0 && vzi < mGridCellsZ
      )
      {

         const QList<uint32_t>& voxel = mGrid[computeVoxelId(vxi, vyi, vzi)];
         __m256i voxelLength = _mm256_set1_epi32(voxel.length());

         if (!voxel.isEmpty())
         {
            linear_cong_gen = (1664525*(particleIndex + almost_a_random) + 1013904223) % 4294967296;
            const int particleOffset = linear_cong_gen % voxel.length();  //rand() % voxel.length();
            almost_a_random++;
            const int particleIterateDirection = (particleIndex % 2) ? -1 : 1;
            __m256i particleOffsetVec = _mm256_set1_epi32(particleOffset);
            __m256i particleIterateDirectionVec = _mm256_set1_epi32(particleIterateDirection);
            __m256i particleId = _mm256_set1_epi32(particleIndex);

            __m256i iii = _mm256_setzero_si256();
            int maxSteps = (voxel.length() + K - 1) / K;

            for (int step = 0; step < maxSteps; ++step)
            {
               //(off + jota) + i * pID
               __m256i nextIndexs = _mm256_add_epi32(particleOffsetVec, jota);
               __m256i nextIndexsaux = _mm256_mullo_epi32(iii, particleIterateDirectionVec);
               nextIndexs = _mm256_add_epi32(nextIndexs, nextIndexsaux);
                              
               __m256i voxelLength = _mm256_set1_epi32(voxel.length());

               __m256i tooSmall = _mm256_cmpgt_epi32(zeros, nextIndexs);           // nextIndexs < 0
               __m256i tooBig = _mm256_cmpgt_epi32(nextIndexs, _mm256_sub_epi32(voxelLength, _mm256_set1_epi32(1)));  // nextIndexs >= voxel.length()

               __m256i invalid = _mm256_or_si256(tooSmall, tooBig);  // invalid = (nextIndexs < 0 || nextIndexs >= voxel.length())

               __m256i valid = _mm256_cmpeq_epi32(invalid, ones);
               int cmp = _mm256_movemask_ps(_mm256_castsi256_ps(invalid));
               __m256i same = _mm256_cmpeq_epi32(nextIndexs, particleId);

               if (cmp != 0)
                  break;

               uint32_t realIndex[K];
               

               alignas(32) int invalidarray[8];
               _mm256_store_si256((__m256i*)invalidarray, invalid);
               alignas(32) int samearray[8];
               _mm256_store_si256((__m256i*)samearray, same);
               alignas(32) int nextIndexsarray[8];
               _mm256_store_si256((__m256i*)nextIndexsarray, nextIndexs);

               #pragma omp simd  // este no se vectoriza :'v
               for (int j = 0; j < K; j++) {
                     int value = invalidarray[j] ? 1 : 0;
                     int same = samearray[j];
                     int idx = nextIndexsarray[j];
                     realIndex[j] = value ? -2 : (same ? -1 : voxel[idx]);
   
               }

               iii = _mm256_add_epi32(iii,ka);

               int validMask[K];
               float dotVals[K];
               int realNeighbors[K];

               #pragma omp simd // este loop se vectoriza bien
               for (int j = 0; j < K; j++) {
                  int idx = realIndex[j];
                  int isValid = (idx >= 0);

                  int base = idx * 3;
                  float dx = pos[0] - mSrcParticles->mPosition[base];
                  float dy = pos[1] - mSrcParticles->mPosition[base + 1];
                  float dz = pos[2] - mSrcParticles->mPosition[base + 2];

                  dx *= isValid;
                  dy *= isValid;
                  dz *= isValid;

                  dotVals[j] = dx * dx + dy * dy + dz * dz;
                  realNeighbors[j] = idx;
                  validMask[j] = isValid;
               }

               __m256 dotValsV = _mm256_loadu_ps(dotVals);
               __m256 cmp1 = _mm256_cmp_ps(dotValsV, mH2vec, _CMP_LT_OQ); // dotVals < mH2

               int simdValid[K];
               #pragma omp simd // este loop se vectoriza bien
               for (int j = 0; j < K; j++) {
                   simdValid[j] = validMask[j] ? 0xFFFFFFFF : 0x00000000;
               }
               __m256 validMaskV = _mm256_castsi256_ps(_mm256_loadu_si256((__m256i*)simdValid));

               __m256 mask = _mm256_and_ps(validMaskV, cmp1);
               int bitmask = _mm256_movemask_ps(mask);

               #pragma omp simd  // este no se vectoriza :'v
               for (int j = 0; j < K; j++) {
                  if (bitmask & (1 << j)) {
                     neighbors[neighborIndex] = realNeighbors[j];
                     neighborDistances[neighborIndex] = sqrtf(dotVals[j]);
                     neighborIndex++;
                  }
               }
            
               enoughNeighborsFound = (neighborIndex > mExamineCount - K);
               if (enoughNeighborsFound){
                  break;
               }
            }
         }
      }
   
         if (enoughNeighborsFound)
            break;
      }

   mSrcParticles->mNeighborCount[particleIndex] = neighborIndex;
}


void SPH::computeDensity(int particleIndex, uint32_t* neighbors, float* neighborDistances)
{
   float density = 0.0f;
   float mass = 0.0f;
   //vec3 pos = mSrcParticles->mPosition[particleIndex];
   float w = 0.0f;
   float rightPart = 0.0f;
   float distanceScaled;

   for (int neighborIndex = 0; neighborIndex < mSrcParticles->mNeighborCount[particleIndex]; neighborIndex++)
   {
      uint32_t realIndex = neighbors[neighborIndex];

      if(realIndex >= mParticleCount)
         break;
      
      if (realIndex != particleIndex)
      {
         // add mass of neighbor
         mass = mSrcParticles->mMass[realIndex];
         // Ya calc las dist en find_neighbors...
         distanceScaled = neighborDistances[neighborIndex];
         // apply smoothing kernel to mass
         if (distanceScaled > mH)
         {
            w = 0.0f;
         }
         else
         {
            // the dot product is not used here since it's not properly scaled
            rightPart = (mH2 - (distanceScaled * distanceScaled));
            rightPart = (rightPart * rightPart * rightPart);
            w = mKernel1Scaled * rightPart;

            // apply weighted neighbor mass to our density
            density += (mass * w);
         }
      }
   }

   mSrcParticles->mDensity[particleIndex] = density;

   // Random: make some have a thounsanth of the density (!)
   /*
   if (particleIndex < M)
   {
      mSrcParticles->mDensity[particleIndex] = density;
   }
   else
   {
      mSrcParticles->mDensity[particleIndex] = density * 1e-3f;
   }
   */
   
}

// NEW: EoS => Algo relacionado a la energía interna (e.g. check lectures Volker)
// Seleccionar (!)
float SPH::computePressure(float rho)
{
   // Option 1: EoS:
   // P = (gamma - 1) * rho * u_int
   // u_int = A * rho^(gamma - 1) * (gamma - 1)^-1
   // Monoatomic gas => gamma = 5/3; A who knows, free parameter...
   //float therm_int = mA_fluid * powf(rho, mGamma_minus) * mInvGamma_minus;
   //return mGamma_minus * rho * therm_int;

   // Option 2: Stiff material:
   return (rho - mRho0) * mStiffness;

}


// De cualquier profile, maybe aca tamb se puede poner la forma del potencial
// Esto es por ahora, NFW estatico...
float SPH::computeEnclosedMassNFW(float dist)
{
   // Necesito los rho_0 y un r_s del perfil (!) Def aca o arriba como const globales

   // NFW:
   // M(< r) = 4 pi rho_0 r_s^3 * [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)] =>
   const float rho_0 = 1e+6;  // Who know que quiero poner aca, recall unidades...
   const float r_s = mH;
   float norm_dist = dist/r_s;

   float inner_term;
   inner_term = logf(1 + norm_dist) - (norm_dist/(1 + norm_dist));

   return 4 * M_PIf * rho_0 * (r_s * r_s * r_s) * inner_term;
}


void SPH::addCentralAccel(float* pos, float* accel)
{
   /*
   <=> Si quiero una accel central

   // New vecs (grav):
   float gravityTerm[3];
   float distance_ij3;
   float rMinusRjScaled[3];

   // Dist al centro
   rMinusRjScaled[0] = (pos[0] - mCentralPos[0]);
   rMinusRjScaled[1] = (pos[1] - mCentralPos[1]);
   rMinusRjScaled[2] = (pos[2] - mCentralPos[2]);

   float dot = (rMinusRjScaled[0] * rMinusRjScaled[0]) + (rMinusRjScaled[1] * rMinusRjScaled[1]) +\
         (rMinusRjScaled[2] * rMinusRjScaled[2]);
   dot = sqrtf(dot);
   distance_ij3 = (dot + mSoftening) * (dot + mSoftening) * (dot + mSoftening);

   gravityTerm[0] = rMinusRjScaled[0]/distance_ij3;
   gravityTerm[1] = rMinusRjScaled[1]/distance_ij3;
   gravityTerm[2] = rMinusRjScaled[2]/distance_ij3;

   // Updateo la gravedad:
   // If "Masa central":
   //accel[0] += -mGravConstant * mCentralMass * gravityTerm[0];
   //accel[1] += -mGravConstant * mCentralMass * gravityTerm[1];
   //accel[2] += -mGravConstant * mCentralMass * gravityTerm[2];

   // If "pot estatico" e.g. NFW:
   // i) F = -grad(Pot_NFW(r))
   // O ii) puedo calcular la masa encerrada a dado radio (simetria esf!!!)
   float enclosed_mass;
   enclosed_mass = computeEnclosedMassNFW(distance_ij3);
   accel[0] += -mGravConstant * enclosed_mass * gravityTerm[0];
   accel[1] += -mGravConstant * enclosed_mass * gravityTerm[1];
   accel[2] += -mGravConstant * enclosed_mass * gravityTerm[2];
   */

   // ---------------------------------------------------------

   // Sino, una accel cte: (lo hago a lo bruto acá, sup gravedad unif hacia abajo (Y-axis))
   accel[1] += -1e+2f;  // Only god knows the correct value for the actual unities
}


void SPH::computeAcceleration(int particleIndex, uint32_t* neighbors, float* neighborDistances)
{
   Particle* neighbor = 0;
   float distanceToNeighborScaled = 0.0f;

   // NEW: EoS (no uso la func...)
   float rho_i = mSrcParticles->mDensity[particleIndex];
   float pi = computePressure(rho_i);

   float rhoiInv = ((rho_i > 0.0f) ? (1.0f / rho_i) : 1.0f);
   float rhoiInv2 = rhoiInv * rhoiInv;
   float piDivRhoi2 = pi * rhoiInv2;

   float r[3];
   r[0] = mSrcParticles->mPosition[particleIndex * 3];
   r[1] = mSrcParticles->mPosition[particleIndex * 3 + 1];
   r[2] = mSrcParticles->mPosition[particleIndex * 3 + 2];

   float vi[3];
   vi[0] = mSrcParticles->mVelocity[particleIndex * 3];
   vi[1] = mSrcParticles->mVelocity[particleIndex * 3 + 1];
   vi[2] = mSrcParticles->mVelocity[particleIndex * 3 + 2];

   float pj = 0.0f;
   float rhoj = 0.0f;
   float rhojInv = 0.0f;
   float rhojInv2 = 0.0f;
   float mj = 0.0f;

   float rj[3];
   float vj[3];
   float rMinusRjScaled[3];

   // pressure gradient...
   float pressureGradient[3] = {0.0f, 0.0f, 0.0f};
   float pressureGradientContribution[3];

   // ...and viscous term
   float viscousTerm[3] = {0.0f, 0.0f, 0.0f};

   // are added to the final acceleration
   float acceleration[3] = {0.0f, 0.0f, 0.0f};

   float centerPart;
   // Acaso llama a cada rato al "neighbor count" o se entiende que es un numero fijo?
   for (int neighborIndex = 0; neighborIndex < mSrcParticles->mNeighborCount[particleIndex]; neighborIndex++)
   {
      uint32_t realIndex = neighbors[neighborIndex];

      rhoj = mSrcParticles->mDensity[realIndex];  // Raro porque puedo re-utilizarlo...
      // NEW, EoS:
      pj = computePressure(rhoj);
      
      rhojInv = ((rhoj > 0.0f) ? (1.0f / rhoj) : 1.0f);
      rhojInv2 = rhojInv * rhojInv;
      //rj = mSrcParticles->mPosition[realIndex];
      rj[0] = mSrcParticles->mPosition[realIndex * 3];
      rj[1] = mSrcParticles->mPosition[realIndex * 3 + 1];
      rj[2] = mSrcParticles->mPosition[realIndex * 3 + 2];
      //vj = mSrcParticles->mVelocity[realIndex];
      vj[0] = mSrcParticles->mVelocity[realIndex * 3];
      vj[1] = mSrcParticles->mVelocity[realIndex * 3 + 1];
      vj[2] = mSrcParticles->mVelocity[realIndex * 3 + 2];

      mj = mSrcParticles->mMass[realIndex];

      // pressure gradient
      rMinusRjScaled[0] = (r[0] - rj[0]);
      rMinusRjScaled[1] = (r[1] - rj[1]);
      rMinusRjScaled[2] = (r[2] - rj[2]);
      distanceToNeighborScaled = neighborDistances[neighborIndex];

      // Ya sabemos que la distancie > 0 (cuando definimos vecinos validos). However,
      // let's add a ~softening
      pressureGradientContribution[0] = mKernel2Scaled * rMinusRjScaled[0] / (distanceToNeighborScaled + 0.01);
      pressureGradientContribution[1] = mKernel2Scaled * rMinusRjScaled[1] / (distanceToNeighborScaled + 0.01);
      pressureGradientContribution[2] = mKernel2Scaled * rMinusRjScaled[2] / (distanceToNeighborScaled + 0.01);

      centerPart = (mH - distanceToNeighborScaled);
      centerPart *= centerPart;
      centerPart *= mj * piDivRhoi2 * (pj * rhojInv2);

      // add pressure gradient contribution to pressure gradient
      pressureGradient[0] += pressureGradientContribution[0] * centerPart;
      pressureGradient[1] += pressureGradientContribution[1] * centerPart;
      pressureGradient[2] += pressureGradientContribution[2] * centerPart;

      // viscosity
      //viscousTermContribution *= (mH - distanceToNeighborScaled); -> "centerpart" (!!!)
      // Reuso variables:
      centerPart = (mH - distanceToNeighborScaled);
      centerPart *= rhojInv * mj * mKernel3Scaled;

      // add contribution to viscous term
      //viscousTerm += viscousTermContribution;
      viscousTerm[0] += (vj[0] - vi[0]) * centerPart;
      viscousTerm[1] += (vj[1] - vi[1]) * centerPart;
      viscousTerm[2] += (vj[2] - vi[2]) * centerPart;

      // Hago acá el viscosityscalar & rho_i^-1
      viscousTerm[0] *= mViscosityScalar * rhoiInv;
      viscousTerm[1] *= mViscosityScalar * rhoiInv;
      viscousTerm[2] *= mViscosityScalar * rhoiInv;

   }
   //viscousTerm *= (mViscosityScalar * rhoiInv);

   //acceleration = viscousTerm - pressureGradient;
   acceleration[0] = viscousTerm[0] - pressureGradient[0];
   acceleration[1] = viscousTerm[1] - pressureGradient[1];
   acceleration[2] = viscousTerm[2] - pressureGradient[2];

   // Updateo la gravedad (select if central mass or static pot):
   addCentralAccel(r, acceleration);  // Void que suma el termino a lo que le das de comer (!)

   // check CFL condition
   float dot;
   dot = (acceleration[0] * acceleration[0]) + (acceleration[1] * acceleration[1]) +\
         (acceleration[2] * acceleration[2]);

   bool limitExceeded = (dot > mCflLimit2);
   if (limitExceeded)
   {
      float length = sqrtf(dot);
      float cflScale = mCflLimit / length;
      acceleration[0] *= cflScale;
      acceleration[1] *= cflScale;
      acceleration[2] *= cflScale;
   }

   mSrcParticles->mAcceleration[particleIndex * 3] = acceleration[0];
   mSrcParticles->mAcceleration[particleIndex * 3 + 1] = acceleration[1];
   mSrcParticles->mAcceleration[particleIndex * 3 + 2] = acceleration[2];
}


// NEW: Cada ~2 steps, checkeá las que tengan alta densidad y que kickeen a
// NO TODOS sus vecinos (!!!)
// *Lo hago acá aparte porque en computeAccel estoy overwritting accels (!)
void SPH::computeFeedback(int particleIndex, uint32_t* neighbors, float* neighborDistances)
{
   // Dada su dens
   float rho_i = mSrcParticles->mDensity[particleIndex];
   // Si es muy alta
   if (rho_i > mRho_thresh)
   {
      float r[3];
      float rj[3];
      float vj[3];
      float direc[3];
      float distanceToNeighborScaled;
      int amount_neighb = mSrcParticles->mNeighborCount[particleIndex];
      // Kickeá a todos tus vecinos (que ya son random!)
      for (int neighborIndex = 0; neighborIndex < amount_neighb; neighborIndex++)
      {
         uint32_t realIndex = neighbors[neighborIndex];

         r[0] = mSrcParticles->mPosition[particleIndex * 3];
         r[1] = mSrcParticles->mPosition[particleIndex * 3 + 1];
         r[2] = mSrcParticles->mPosition[particleIndex * 3 + 2];

         rj[0] = mSrcParticles->mPosition[realIndex * 3];
         rj[1] = mSrcParticles->mPosition[realIndex * 3 + 1];
         rj[2] = mSrcParticles->mPosition[realIndex * 3 + 2];

         distanceToNeighborScaled = neighborDistances[neighborIndex];
         direc[0] = (r[0] - rj[0])/distanceToNeighborScaled;
         direc[1] = (r[1] - rj[1])/distanceToNeighborScaled;
         direc[2] = (r[2] - rj[2])/distanceToNeighborScaled;

         // Updateá la velocidad con un kick (¿Sirve?)
         mSrcParticles->mVelocity[realIndex * 3] += direc[0] * mKickVel/amount_neighb;
         mSrcParticles->mVelocity[realIndex * 3 + 1] += direc[1] * mKickVel/amount_neighb;
         mSrcParticles->mVelocity[realIndex * 3 + 2] += direc[2] * mKickVel/amount_neighb;

      }
   }

}


void SPH::integrate(int particleIndex)
{
   // LF-KDK: Only gravity:
   float position[3];
   position[0] = mSrcParticles->mPosition[particleIndex * 3];
   position[1] = mSrcParticles->mPosition[particleIndex * 3 + 1];
   position[2] = mSrcParticles->mPosition[particleIndex * 3 + 2];
   
   float velocity[3];
   velocity[0] = mSrcParticles->mVelocity[particleIndex * 3];
   velocity[1] = mSrcParticles->mVelocity[particleIndex * 3 + 1];
   velocity[2] = mSrcParticles->mVelocity[particleIndex * 3 + 2];
   
   float acceleration[3];
   acceleration[0] = mSrcParticles->mAcceleration[particleIndex * 3];
   acceleration[1] = mSrcParticles->mAcceleration[particleIndex * 3 + 1];
   acceleration[2] = mSrcParticles->mAcceleration[particleIndex * 3 + 2];
   //float mass_here = mSrcParticles->mMass[particleIndex];

   float velocity_halfstep[3];
   velocity_halfstep[0] = velocity[0] + (acceleration[0] * mTimeStep * 0.5f);
   velocity_halfstep[1] = velocity[1] + (acceleration[1] * mTimeStep * 0.5f);
   velocity_halfstep[2] = velocity[2] + (acceleration[2] * mTimeStep * 0.5f);

   float newPosition[3];
   newPosition[0] = position[0] + (velocity_halfstep[0] * mTimeStep);
   newPosition[1] = position[1] + (velocity_halfstep[1] * mTimeStep);
   newPosition[2] = position[2] + (velocity_halfstep[2] * mTimeStep);

   // Copy & paste grav -> Tengo la func:
   addCentralAccel(newPosition, acceleration);
   
   float newVelocity[3];
   newVelocity[0] = velocity_halfstep[0] + (acceleration[0] * mTimeStep);
   newVelocity[1] = velocity_halfstep[1] + (acceleration[1] * mTimeStep);
   newVelocity[2] = velocity_halfstep[2] + (acceleration[2] * mTimeStep);

   // Muchos NaNs... Skip them:
   /* if (dot > 0)
   {
      // Calc acá T, W y L del sistema (no guardar las energías en las Particles)
      #pragma omp atomic
      mKineticEnergyTotal += 0.5f * mass_here * dot;

      // Energía potencial sería G * Mcentral * m_i/r_i
      #pragma omp atomic
      mPotentialEnergyTotal -= mGravConstant * mCentralMass * mass_here / distance_ij3;
      // + softening);  // B: Without soft (i.e. without a Plummer equivalent)

      // WIP
      //mAngularMomentumTotal += (mass_here * (newPosition - mCentralPos).cross(newVelocity));

   } */

   // Antes de terminar la integración: Si te pasate del borde -> rebotá
   updateBoundary(newPosition, newVelocity);

   // Ahora si, actualizá
   mSrcParticles->mPosition[particleIndex * 3] = newPosition[0];
   mSrcParticles->mPosition[particleIndex * 3 + 1] = newPosition[1];
   mSrcParticles->mPosition[particleIndex * 3 + 2] = newPosition[2];

   mSrcParticles->mVelocity[particleIndex * 3] = newVelocity[0];
   mSrcParticles->mVelocity[particleIndex * 3 + 1] = newVelocity[1];
   mSrcParticles->mVelocity[particleIndex * 3 + 2] = newVelocity[2];
}


// Try mine, a ver que onda:
// Check for position. If past the wall -> bounce (reflexion on velocity)
// Recall que el "piso" es en el eje Y...
void SPH::updateBoundary(float* pos, float* vel)
{
   // X-axis (vel para evitar que reboten inf):
   if (pos[0] < 0.f && vel[0] < 0.f)
   {
      vel[0] *= -mDamping;
   }
   else if (pos[0] > mMaxX && vel[0] > 0.f)
   {
      vel[0] *= -mDamping;
   }

   // Y-axis:
   if (pos[1] < 0.f && vel[1] < 0.f)
   {
      vel[1] *= -mDamping;
   }
   else if (pos[1] > mMaxY && vel[1] > 0.f)
   {
      vel[1] *= -mDamping;
   }

   // Z-axis:
   if (pos[2] < 0.f && vel[2] < 0.f)
   {
      vel[2] *= -mDamping;
   }
   else if (pos[2] > mMaxZ && vel[2] > 0.f)
   {
      vel[2] *= -mDamping;
   }
}

/*
Old

void SPH::handleBoundaryConditions(
   vec3 position,
   vec3* newVelocity,
   float timeStep,
   vec3* newPosition
)
{
   // x coord
   if (newPosition->x < 0.0f)
   {
      vec3 normal(1, 0, 0);
      float intersectionDistance = -position.x / newVelocity->x;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
   else if (newPosition->x > mMaxX)
   {
      vec3 normal(-1, 0, 0);
      float intersectionDistance = (mMaxX - position.x) / newVelocity->x;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }

   // y coord
   if (newPosition->y < 0.0f)
   {
      vec3 normal(0, 1, 0);
      float intersectionDistance = -position.y / newVelocity->y;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
   else if (newPosition->y > mMaxY)
   {
      vec3 normal(0, -1, 0);
      float intersectionDistance = (mMaxY - position.y) / newVelocity->y;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }

   // z coord
   if (newPosition->z < 0.0f)
   {
      vec3 normal(0, 0, 1);
      float intersectionDistance = -position.z / newVelocity->z;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
   else if (newPosition->z > mMaxZ)
   {
      vec3 normal(0, 0, -1);
      float intersectionDistance = (mMaxZ - position.z) / newVelocity->z;

      applyBoundary(
         position,
         timeStep,
         newPosition,
         intersectionDistance,
         normal,
         newVelocity
      );
   }
}


void SPH::applyBoundary(
      vec3 position,
      float timeStep,
      vec3* newPosition,
      float intersectionDistance,
   vec3 normal,
   vec3* newVelocity
)
{
   
   vec3 intersection = position + (*newVelocity * intersectionDistance);

   float dotProduct =
        newVelocity->x * normal.x
      + newVelocity->y * normal.y
      + newVelocity->z * normal.z;

   vec3 reflection = *newVelocity - (normal * dotProduct * 2.0f);

   float remaining = timeStep - intersectionDistance;

   // apply boundaries
   *newVelocity = reflection;
   *newPosition = intersection + reflection * (remaining * mDamping);
}
*/


int SPH::computeVoxelId(int voxelX, int voxelY, int voxelZ)
{
   return (voxelZ * mGridCellsY + voxelY) * mGridCellsX + voxelX;
}


void SPH::clearNeighbors()
{
   memClear32(mNeighbors, mParticleCount * mExamineCount * sizeof(Particle*));
}


void SPH::memClear32(void* dst, int len)
{
   unsigned int* dst32= (unsigned int*)dst;
   len>>=2;
   while (len--)
      *dst32++= 0;
}


float SPH::getCellSize() const
{
   return mCellSize;
}


Particle* SPH::getParticles()
{
   return mSrcParticles;
}


int SPH::getParticleCount() const
{
   return mParticleCount;
}


void SPH::getGridCellCounts(int &x, int &y, int &z)
{
   x = mGridCellsX;
   y = mGridCellsY;
   z = mGridCellsZ;
}


void SPH::getParticleBounds(float &x, float &y, float &z)
{
   x = mMaxX;
   y = mMaxY;
   z = mMaxZ;
}


float SPH::getInteractionRadius2() const
{
   return mH2;
}


QList<uint32_t>* SPH::getGrid()
{
   return mGrid;
}



vec3 SPH::getGravity() const
{
   return mGravity;
}


void SPH::setGravity(const vec3 &gravity)
{
   mGravity = gravity;
}


float SPH::getCflLimit() const
{
   return mCflLimit;
}


void SPH::setCflLimit(float cflLimit)
{
   mCflLimit = cflLimit;
   mCflLimit2 = mCflLimit * mCflLimit;
}


float SPH::getDamping() const
{
   return mDamping;
}


void SPH::setDamping(float damping)
{
   mDamping = damping;
}


float SPH::getTimeStep() const
{
   return mTimeStep;
}


void SPH::setTimeStep(float timeStep)
{
   mTimeStep = timeStep;
}


float SPH::getViscosityScalar() const
{
   return mViscosityScalar;
}


void SPH::setViscosityScalar(float viscosityScalar)
{
   mViscosityScalar = viscosityScalar;
}


float SPH::getStiffness() const
{
   return mStiffness;
}


void SPH::setStiffness(float stiffness)
{
   mStiffness = stiffness;
}
