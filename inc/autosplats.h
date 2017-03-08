#pragma once

// GL graphics includes
#ifdef _WIN32
  #include <GL/glew.h>
  #include <GL/wglew.h>
#endif

// other includes
#include <vector>
#include <stdio.h>
#include <string.h>
using namespace std;

#include "vec.hpp"


namespace gms
{
	class AutoSplats
	{
	// parameters
	public:
		struct Params
		{
      Params() :splatRadiusFactor(1) {}

			uint K;
			uint maxIters;
			bool rs16bit;
			bool instantGather;
			bool projectToMean;
			float splatRadiusFactor;
		};

	public:
		struct Timings {
			double project;
			double collectPoints;
			double initRadius;
			double rsDist;
			double rsGather;
			double finalDist;
			double hcov;
			double computeSplats;
			double splatting;
			double total;
		} timings;

	public:
		AutoSplats(const string& shaderDir);
		~AutoSplats();

		void resize(uint w, uint h, const Params& params);
    void render(uint vboPoints, uint vboNormals, uint numPoints, const mat4& modelViewMatrix, const mat4& projMatrix, float zNear, float zFar, const Params& params);

		uint getASPointCount() const	{ return mASPointCount; }
		uint getASPositions() const		{ return mASPositionVbo; }
		uint getASNormals() const		{ return mASNormalsVbo; }
		uint getASSplatAxes() const		{ return mASSplatAxesVbo; }


	private:
		void drawPoints(uint vbo, int numComponents, uint glType, uint numPoints);
		void drawScreenPoints();
    void initialProjection(uint vboPoints, uint vboNormals, uint numPoints);
		void createScreenPointVBO();
		void initSearchRadiusTexture(const Params& params);
		void adaptSearchRadii(int iter, float zNear, float zFar, const Params& params);
		void adaptSearchRadiiInstantGather(int iter, float zNear, float zFar, const Params& params);
		void finalDistribute(float zNear, float zFar);
		void gatherHCov(float zNear, float zFar);
		void computeSplatVBOs(const Params& params);
  
  public:
    void surfaceSplatting(const mat4& modelViewMatrix, const mat4& projMatrix, float zNear, float zFar, const Params& params);
				
	public:
		bool mRs16bit;
		vector<uint> mShaders;

		mat4 mModelViewMatrix;
		mat4 mProjMatrix;
		mat4 mModelViewProjMatrix;
		vec2 mViewPlaneRT;
		vec2 mInvViewPlaneRT;
		float mVPixelDiagonal;
		float mZNear;
		vec2 mInvVpSize;

		// constants
		uint mWidth;
		uint mHeight;
		uint mWidthCoarse;
		uint mHeightCoarse;

		// shader programs
		struct {
			uint initialProjection;
			uint createPixelVertexVBO;
			uint projectCounterGrid;
			uint initSearchRadiusGrid;
			uint rangeSearchDistribute;
			uint rangeSearchGather;
			uint rangeSearchInstantGather;
			uint finalDistribute;
			uint gatherHCov;
			uint computeSplatVBOs;
			uint splatDepth;
			uint splatAccum;
			uint splatFinal;
			uint texture;
		} mProg;
		
		// queries
		uint mScreenPointCountQuery;

		// vbos
		uint mVboPixelVertices;
		uint mVboScreenPoints;

  public:
		uint mNumScreenPoints;


		// fbos and rbos
		uint mFboScreenAS;
		uint mFboScreenSS;
		uint mFboCoarseCnt;

		uint mRboScreenAS;
		uint mRboScreenSS;

		// OUTPUT
    public:
		uint mASPositionVbo;
		uint mASNormalsVbo;
		uint mASSplatAxesVbo;
    uint mASCorrectNormalsVbo;

		uint mASPointCount;

	public:
		struct Textures {
			uint pointPos;
			uint counterGrid;
			uint sr0;
			uint sr1;
			uint neighCount;
			uint fr;
			uint frFinal;
			uint hcovMean;
			uint hcov1;
			uint hcov2;
			uint depth;
			uint accumNormalDepth;
			uint accumColorWeight;
      uint correctNormals;
		};
		const Textures& getTextures() const { return mTex; }
		int getTexCount() const { return sizeof(Textures) / sizeof(uint); }

	public:
		// textures
		Textures mTex;

    // fullscreen vertices
    GLuint quadVbo;


	// TODO: Delete - deprecated?
	public:
		const uint* getNumScreenPointsPtr() const { return &mNumScreenPoints; }
	};
}

