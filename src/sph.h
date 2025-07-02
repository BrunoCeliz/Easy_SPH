#ifndef SPH_H
#define SPH_H

// Qt
#include <QList>
#include <QMutex>
#include <QThread>

// sph
#include "vec3.h"
#include "vec3i.h"

class Particle;

class SPH : public QThread
{

Q_OBJECT

   public:

      SPH();
      virtual ~SPH();

      // control
      bool isStopped() const;

      bool isPaused() const;

      // getters and setters
      Particle* getParticles();

      int getParticleCount() const;

      void getGridCellCounts(int& x, int& y, int& z);

      void getParticleBounds(float& x, float&y, float& z);

      float getInteractionRadius2() const;

      QList<uint32_t>* getGrid();

      float getCellSize() const;

      vec3 getGravity() const;
      void setGravity(const vec3 &gravity);

      float getStiffness() const;
      void setStiffness(float stiffness);

      float getViscosityScalar() const;
      void setViscosityScalar(float viscosityScalar);

      float getTimeStep() const;
      void setTimeStep(float timeStep);

      float getDamping() const;
      void setDamping(float damping);

      float getCflLimit() const;
      void setCflLimit(float cflLimit);


public slots:

      void run();
      void step(int this_step);

      void pauseResume();
      void stopSimulation();


signals:

      void updateElapsed(
         int,
         int,
         int,
         int,
         int,
         int
      );

      void stepFinished();


protected:

      void initParticlePositionsRandom();

      void initParticlePolitionsSphere();

      void clearGrid();

      // voxelize
      void voxelizeParticles();

      // neighbor localization
      void findNeighbors(
         int particleIndex,
         uint32_t* neighbors,
         int voxelX,
         int voxelY,
         int voxelZ,
         float* distances
      );

      // physics
      void computeDensity(int p, uint32_t* neighbors, float* distances);
      float computePressure(float rho);
      void computeAcceleration(int p, uint32_t* neighbors, float* distances);
      // New:
      void computeFeedback(int p, uint32_t* neighbors, float* distances);
      // New para central grav:
      float computeEnclosedMassNFW(float dist);
      void addCentralAccel(float* pos, float* accel);

      void integrate(int p);


      // helper functions
      int computeVoxelId(int voxelX, int voxelY, int voxelZ);

      void applyBoundary(
         vec3 position,
         float getTimeStep,
         vec3 *newPosition,
         float intersectionDistance,
         vec3 normal,
         vec3 *newVelocity
      );

      void handleBoundaryConditions(
         vec3 position,
         vec3 *newVelocity,
         float getTimeStep,
         vec3 *newPosition
      );


      void clearNeighbors();
      void memClear32(void *dst, int len);

      // grid related
      Particle* mSrcParticles;
      int* mVoxelIds;
      vec3i* mVoxelCoords;

      unsigned int* mParticleIndices;

      // ints de 32
      int mParticleCount;
      int mGridCellsX;
      int mGridCellsY;
      int mGridCellsZ;
      int mGridCellCount;

      float mCellSize;

      float mMaxX;
      float mMaxY;
      float mMaxZ;

      int mExamineCount;

      float mH;
      float mH2;
      float mH6;
      float mH9;
      float mHTimes2;
      float mHTimes2Inv;

      QList<uint32_t>* mGrid;
      uint32_t* mNeighbors;
      float* mNeighborDistancesScaled;

      int totalSteps;
      //float mKineticEnergyTotal;
      //float mPotentialEnergyTotal;
      //vec3 mAngularMomentumTotal;

      // timers
      int timeVoxelize;
      int timeFindNeighbors;
      int timeComputeDensity;
      int timeComputePressure;
      int timeComputeAcceleration;
      int timeIntegrate;

      // physics
      float mRho0;
      float mStiffness;
      vec3 mGravity;
      float mKernel1Scaled;
      float mKernel2Scaled;
      float mKernel3Scaled;
      float mViscosityScalar; // actually 'mu'
      float mTimeStep;
      float mDamping;
      float mCflLimit;
      float mCflLimit2;
      // Gravity constant:
      float mGravConstant;
      // Central mass (and pos):
      float mCentralMass;
      //vec3 mCentralPos;
      float mCentralPos[3];
      // Softening (force)
      float mSoftening;

      // NEW: EoS
      float mA_fluid;
      float mGamma_minus;
      float mInvGamma_minus;

      // Feedback
      float mRho_thresh;
      float mKickVel;

      // thread handling
      mutable QMutex mMutex;
      bool mStopped;
      bool mPaused;
};

#endif // SPH_H


