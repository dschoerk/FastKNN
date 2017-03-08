#include "autosplats.h"
#include "glbase.hpp"
#include <iostream>
using namespace std;


namespace gms
{

	AutoSplats::AutoSplats(const string& shaderDir)
	{
		string dir = shaderDir + "/";

		int fsRT0 = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "rt0.frag");
		int fsRT01 = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "rt01.frag");
		int vsProjectCounterGrid = loadShader(mShaders, GL_VERTEX_SHADER, dir + "projectCounterGrid.vert");
		int vsProgInitialProjection = loadShader(mShaders, GL_VERTEX_SHADER, dir + "initialProjection.vert");
		int vsPixelVertices = loadShader(mShaders, GL_VERTEX_SHADER, dir + "pixelVertices.vert");
		int gsScreenPosSO = loadShader(mShaders, GL_GEOMETRY_SHADER, dir + "screenPosSO.geom");
		int vsInitSearchRadiusGrid = loadShader(mShaders, GL_VERTEX_SHADER, dir + "initSearchRadiusGrid.vert");
		int gsProjectSphere = loadShader(mShaders, GL_GEOMETRY_SHADER, dir + "projectSphere.geom");
		int vsRangeSearchDistribute = loadShader(mShaders, GL_VERTEX_SHADER, dir + "rangeSearchDistribute.vert");
		int fsRangeSearchDistribute = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "rangeSearchDistribute.frag");
		int vsRangeSearchGather = loadShader(mShaders, GL_VERTEX_SHADER, dir + "rangeSearchGather.vert");
		int fsRangeSearchGather = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "rangeSearchGather.frag");
		int vsRangeSearch2 = loadShader(mShaders, GL_VERTEX_SHADER, dir + "rangeSearchInstantGather.vert");
		int fsRangeSearch2 = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "rangeSearchInstantGather.frag");
		int vsFinalDistribute = loadShader(mShaders, GL_VERTEX_SHADER, dir + "finalDistribute.vert");
		int fsFinalDistribute = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "finalDistribute.frag");
		int vsGatherHCov = loadShader(mShaders, GL_VERTEX_SHADER, dir + "gatherHCov.vert");
		int fsGatherHCov = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "gatherHCov.frag");
		int vsComputeSplatVbos = loadShader(mShaders, GL_VERTEX_SHADER, dir + "computeSplatVbos.vert");
		int gsComputeSplatVbosSO = loadShader(mShaders, GL_GEOMETRY_SHADER, dir + "computeSplatVbosSO.geom");
		int gsSplat = loadShader(GL_GEOMETRY_SHADER, dir + "splatDepth.geom");
		int vsSplatDepth = loadShader(mShaders, GL_VERTEX_SHADER, dir + "splatDepth.vert");
		int fsSplatDepth = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "splatDepth.frag");
		int vsSplatAccum = loadShader(mShaders, GL_VERTEX_SHADER, dir + "splatAccum.vert");
		int fsSplatAccum = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "splatAccum.frag");
		int vsSplatFinal = loadShader(mShaders, GL_VERTEX_SHADER, dir + "splatFinal.vert");
		int fsSplatFinal = loadShader(mShaders, GL_FRAGMENT_SHADER, dir + "splatFinal.frag");

		
		mProg.initialProjection = createProgram(vsProgInitialProjection, 0, fsRT0, "InitialProjection");
		mProg.createPixelVertexVBO = createProgram(vsPixelVertices, gsScreenPosSO, 0, "CreatePixelVertexVBO");
		mProg.projectCounterGrid = createProgram(vsProjectCounterGrid, 0, fsRT0, "ProjectCounterGrid");
		mProg.initSearchRadiusGrid = createProgram(vsInitSearchRadiusGrid, 0, fsRT01, "InitSearchRadiusGrid");
		mProg.rangeSearchDistribute = createProgram(vsRangeSearchDistribute, gsProjectSphere, fsRangeSearchDistribute, "RangeSearchDistribute");
		mProg.rangeSearchGather = createProgram(vsRangeSearchGather, gsProjectSphere, fsRangeSearchGather, "RangeSearchGather");
		mProg.rangeSearchInstantGather = createProgram(vsRangeSearch2, gsProjectSphere, fsRangeSearch2, "RangeSearchInstantGather");
		mProg.finalDistribute = createProgram(vsFinalDistribute, gsProjectSphere, fsFinalDistribute, "FinalDistribute");
		mProg.gatherHCov = createProgram(vsGatherHCov, gsProjectSphere, fsGatherHCov, "GatherHCov");
		mProg.computeSplatVBOs = createProgram(vsComputeSplatVbos, gsComputeSplatVbosSO, 0, "ComputeSplatVBOs");
		mProg.splatDepth = createProgram(vsSplatDepth, gsSplat, fsSplatDepth, "SplatDepth");
		mProg.splatAccum = createProgram(vsSplatAccum, gsSplat, fsSplatAccum, "SplatAccum");
		mProg.splatFinal = createProgram(vsSplatFinal, 0, fsSplatFinal, "SplatFinal");


		const char *varyings[] = { "so_uvPos" };
		glTransformFeedbackVaryings(mProg.createPixelVertexVBO, 1, varyings, GL_SEPARATE_ATTRIBS);
		glLinkProgram(mProg.createPixelVertexVBO);

		const char *varyings2[] = { "so0", "so1", "so2", "so3" };
		glTransformFeedbackVaryings(mProg.computeSplatVBOs, 4, varyings2, GL_SEPARATE_ATTRIBS);
		glLinkProgram(mProg.computeSplatVBOs);

		// Check Transform Feedback Linking
		uint prog = mProg.computeSplatVBOs;
		{
			int success = 0;
			glGetProgramiv(prog, GL_LINK_STATUS, &success);
			if (!success) {
				char info[1024] = { 0 };
				glGetProgramInfoLog(prog, 1024, 0, info);
				std::cout << info << std::endl;
			}
		}


		// Queries
		glGenQueries(1, &mScreenPointCountQuery);

		// Vbos
		glGenBuffers(2, &mVboPixelVertices);

		// Output Vbos
		glGenBuffers(4, &mASPositionVbo);
		mASPointCount = 0;
		
		// FBOs and Textures
		glGenFramebuffers(3, &mFboScreenAS);
		glGenTextures(getTexCount(), (uint*)&mTex);
		glGenRenderbuffers(2, &mRboScreenAS);
		
		// Coarse Grid Counting Texture
		glBindFramebuffer(GL_FRAMEBUFFER, mFboCoarseCnt);
		attachTexture(0, mTex.counterGrid);

		// Autosplats Screen FBO and RBO
		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
		glBindRenderbuffer(GL_RENDERBUFFER, mRboScreenAS);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1, 1);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRboScreenAS);
		attachTexture(0, mTex.pointPos);
		attachTexture(1, mTex.sr0);
		attachTexture(2, mTex.sr1);
		attachTexture(3, mTex.fr);
		attachTexture(4, mTex.frFinal);
		attachTexture(5, mTex.hcovMean);
		attachTexture(6, mTex.hcov1);
		attachTexture(7, mTex.hcov2);
    attachTexture(1, mTex.correctNormals);

		
		uint fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
			cout << "Error creating FboScreenAS! (" << fboStatus << ")" << endl;


		// Surface Splatting Screen FBO and RBO
		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenSS);
		glBindRenderbuffer(GL_RENDERBUFFER, mRboScreenSS);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1, 1);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mRboScreenSS);
		attachTexture(0, mTex.depth);
		attachTexture(1, mTex.accumNormalDepth);
		attachTexture(2, mTex.accumColorWeight);
		
		fboStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (fboStatus != GL_FRAMEBUFFER_COMPLETE)
			cout << "Error creating FboScreenSS! (" << fboStatus << ")" << endl;
		

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GLfloat vertices[] = { -1.f, -1.f, 1.f, -1.f, -1.f, 1.f, -1.f, 1.f, 1.f, -1.f, 1.f, 1.f };
    glGenBuffers(1, &quadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
	}


	AutoSplats::~AutoSplats()
	{
		// delete programs
		for (uint* pProg = (uint*)&mProg; pProg < (uint*)&mProg + sizeof(mProg) / sizeof(uint); pProg++)
			glDeleteProgram(*pProg);

		// delete shaders
    for (unsigned int i = 0; i < mShaders.size(); ++i)
      glDeleteShader(mShaders[i]);
		
		// delete fbos, rbos, textures
		glDeleteFramebuffers(3, &mFboScreenAS);
		glDeleteTextures(getTexCount(), (uint*)&mTex);
		glDeleteRenderbuffers(2, &mRboScreenAS);
	}



	void AutoSplats::resize(uint w, uint h, const Params& params)
	{
		mWidth = w;
		mHeight = h;
		mInvVpSize = vec2(1.0f / w, 1.0f / h);
		
		// Resize Coarse Fbo
		glBindTexture(GL_TEXTURE_2D, mTex.counterGrid);	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, mWidth, mHeight, 0, GL_RED, GL_FLOAT, NULL);
		
		int rsPrecision = params.rs16bit ? GL_RGBA16F : GL_RGBA32F;

		// AS Screen Fbo
		glBindRenderbuffer(GL_RENDERBUFFER, mRboScreenAS);	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mWidth, mHeight);
		glBindTexture(GL_TEXTURE_2D, mTex.pointPos);		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, mTex.correctNormals);		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, mWidth, mHeight, 0, GL_RGB, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.sr0);				glTexImage2D(GL_TEXTURE_2D, 0, rsPrecision, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.sr1);				glTexImage2D(GL_TEXTURE_2D, 0, rsPrecision, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.fr);				glTexImage2D(GL_TEXTURE_2D, 0, rsPrecision, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.frFinal);			glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.hcovMean);		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.hcov1);			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.hcov2);			glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.neighCount);		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, mWidth, mHeight, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, NULL);
		
		// SS Screen Fbo
		glBindRenderbuffer(GL_RENDERBUFFER, mRboScreenSS);		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, mWidth, mHeight);
		glBindTexture(GL_TEXTURE_2D, mTex.depth);				glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);		// TEMP: RG
		glBindTexture(GL_TEXTURE_2D, mTex.accumNormalDepth);	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		glBindTexture(GL_TEXTURE_2D, mTex.accumColorWeight);	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mWidth, mHeight, 0, GL_RGBA, GL_FLOAT, NULL);
		
		// Screen Pixel VBO
		glBindBuffer(GL_ARRAY_BUFFER, mVboPixelVertices);	glBufferData(GL_ARRAY_BUFFER, 1 * mWidth * mHeight, 0, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, mVboScreenPoints);	glBufferData(GL_ARRAY_BUFFER, mWidth * mHeight * 2 * sizeof(float), 0, GL_STATIC_DRAW);

		// Output VBOs
		glBindBuffer(GL_ARRAY_BUFFER, mASPositionVbo);		glBufferData(GL_ARRAY_BUFFER, mWidth * mHeight * 3 * sizeof(float), 0, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, mASNormalsVbo);		glBufferData(GL_ARRAY_BUFFER, mWidth * mHeight * 3 * sizeof(float), 0, GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, mASSplatAxesVbo);	glBufferData(GL_ARRAY_BUFFER, mWidth * mHeight * 4 * sizeof(float), 0, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, mASCorrectNormalsVbo);	glBufferData(GL_ARRAY_BUFFER, mWidth * mHeight * 3 * sizeof(float), 0, GL_STATIC_DRAW);
		
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}




	// takes VBO of input points (vboPoints) and computes splats stored in splat vbos
	void AutoSplats::render(uint vboPoints, uint vboNormals, uint numPoints, const mat4& modelViewMatrix, const mat4& projMatrix, float zNear, float zFar, const Params& params)
	{
		// update buffers if precision changed
		if (mRs16bit != params.rs16bit)
		{
			mRs16bit = params.rs16bit;
			resize(mWidth, mHeight, params);
		}
		
		memset(&timings, 0, sizeof(timings));
		double frameStart = getTime();

		mModelViewMatrix = modelViewMatrix;
		mProjMatrix = projMatrix;
		mModelViewProjMatrix = mProjMatrix * mModelViewMatrix;
		mViewPlaneRT = vec2(zNear / mProjMatrix(0, 0), zNear / mProjMatrix(1, 1));
		mInvViewPlaneRT = 1.0 / mViewPlaneRT;
		mZNear = zNear;
		float pixelWidth = mViewPlaneRT.x / mWidth;
		float pixelHeight = mViewPlaneRT.y / mHeight;
		mVPixelDiagonal = sqrtf(pixelWidth*pixelWidth + pixelHeight*pixelHeight);
		
		//-------------------------------------------------------
		glViewport(0, 0, mWidth, mHeight);
		glClearColor(0, 0, 0, 0);
		glClearDepth(1.0f);
		//-------------------------------------------------------


		// 1. Initial Projection to wPos texture
		startTimer();
    initialProjection(vboPoints, vboNormals, numPoints);
		timings.project += stopTimer();
    //return;

		// 2. Let screen pixels lookup remaining points ad create uv-VBO
		startTimer();
		createScreenPointVBO();
		timings.collectPoints += stopTimer();

		// 3. create coarse counter grid points and initialize radii
		startTimer();
		initSearchRadiusTexture(params);
		timings.initRadius += stopTimer();


		// 4. search knns
		//memset( mUnconvergedPointCount, 0, MAX_ITERATIONS * sizeof(UINT32) );
		for (uint iter = 0; iter < params.maxIters; ++iter)
		{
			if (params.instantGather)
			{
				startTimer();
				adaptSearchRadiiInstantGather(iter, zNear, zFar, params);
				timings.rsDist += stopTimer();
			}
			else
			{
				adaptSearchRadii(iter, zNear, zFar, params);
			}
		}
    //return;
		
		// 5. compute splats
		startTimer();
		finalDistribute(zNear, zFar);
		timings.finalDist = stopTimer();

		// 6. feedback hcov
		startTimer();
		gatherHCov(zNear, zFar);
		timings.hcov = stopTimer();

		// 7. compute splat VBOs
		startTimer();
		computeSplatVBOs(params);
		timings.computeSplats = stopTimer();

		// now we have to reset depth mask and depth comparison mode for normal rendering!
		glDepthMask(true);
		glDepthFunc(GL_LESS);
		
		// 8. perform surface splatting
		startTimer();
		surfaceSplatting(modelViewMatrix, projMatrix, zNear, zFar, params);
		timings.splatting = stopTimer();

				
		glFinish();
		timings.total = getTime() - frameStart;
	}



	void AutoSplats::initialProjection(uint vboPoints, uint vboNormals, uint numPoints)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 1, GL_TEXTURE_2D, mTex.correctNormals, 0);

    setDrawBuffers(0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		glUseProgram(mProg.initialProjection);
		glUniformMatrix4fv(glGetUniformLocation(mProg.initialProjection, "mvpMatrix"), 1, true, mModelViewProjMatrix);

		//drawPoints(vboPoints, 3, GL_FLOAT, numPoints);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vboPoints);
    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);
    
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, vboNormals);
    glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, 0);
    
    glDrawArrays(GL_POINTS, 0, numPoints);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

		glUseProgram(0);
		


/*    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, mFboScreenAS);
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glBlitFramebuffer(0, 0, 1024, 1024, 0, 0, 1024, 1024, GL_COLOR_BUFFER_BIT, GL_NEAREST);*/

    //attachTexture(1, );
    //glBindTexture(GL_TEXTURE_2D, mTex.sr0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 1, GL_TEXTURE_2D, mTex.sr0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);
	}



	/// writes screen points to vbo and at the same time creates their proper depth footprint
	void AutoSplats::createScreenPointVBO()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
		glDrawBuffer(GL_NONE);			// draws only to depth buffer
		glEnable(GL_DEPTH_TEST);
		glClearDepth(1);		// Switch this to 0 to see the splats for debugging
		glDepthFunc(GL_ALWAYS);
		glClear(GL_DEPTH_BUFFER_BIT);

		glUseProgram(mProg.createPixelVertexVBO);
		glUniformMatrix4fv(glGetUniformLocation(mProg.createPixelVertexVBO, "mvpMatrix"), 1, true, mModelViewProjMatrix);
		glUniform2f(glGetUniformLocation(mProg.createPixelVertexVBO, "invVpSize"), mInvVpSize.x, mInvVpSize.y);
		bindUniformSampler(mProg.createPixelVertexVBO, 0, "texWPos", mTex.pointPos);

		// Transform Feedback Initialization
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, mVboScreenPoints);
		glBeginTransformFeedback(GL_POINTS);
		glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, mScreenPointCountQuery);

		drawPoints(mVboPixelVertices, 1, GL_UNSIGNED_BYTE, mWidth * mHeight);

		// End Transform Feedback
		glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
		glEndTransformFeedback();
		mNumScreenPoints = 0;
		glGetQueryObjectuiv(mScreenPointCountQuery, GL_QUERY_RESULT, &mNumScreenPoints);		// read back query results

		// reset states
		glUseProgram(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);	glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glClearDepth(1);
	}



	void AutoSplats::initSearchRadiusTexture(const Params& params)
	{
		// PASS 1:
		//-------------------------------------
		// adaptively determine the ideal grid coarseness, based on global avg screen density
		float cellScaleFactor = 1.0f;

		float cellSideLength = fmaxf(cellScaleFactor * sqrtf(mWidth * mHeight * params.K / float(max(mNumScreenPoints, 1u))), 1.0f);
		mWidthCoarse = uint(mWidth / cellSideLength + 1);
		mHeightCoarse = uint(mHeight / cellSideLength + 1);
		float stretchX = mWidth / ((float)mWidthCoarse * cellSideLength);
		float stretchY = mHeight / ((float)mHeightCoarse * cellSideLength);


		glBindFramebuffer(GL_FRAMEBUFFER, mFboCoarseCnt);
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, mWidthCoarse, mHeightCoarse);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		glUseProgram(mProg.projectCounterGrid);
		glUniform2f(glGetUniformLocation(mProg.projectCounterGrid, "stretchFactor"), stretchX, stretchY);

		drawScreenPoints();

		glDisable(GL_BLEND);


		// PASS 2:
		//-------------------------------------
		// from now on we disable depth writes to conserve the current depth footprint, and we flip the depth test operaor
		glDepthMask(false);
		glDepthFunc(GL_GREATER);
		glDisable(GL_DEPTH_TEST);	// important - we want to write all points at initialization


		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
		setDrawBuffers(1, 2);
		glViewport(0, 0, mWidth, mHeight);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(mProg.initSearchRadiusGrid);
		glUniformMatrix4fv(glGetUniformLocation(mProg.initSearchRadiusGrid, "viewMatrix"), 1, true, mModelViewMatrix);
		glUniform2f(glGetUniformLocation(mProg.initSearchRadiusGrid, "invVpSize"), 1.0f / mWidth, 1.0f / mHeight);
		glUniform2f(glGetUniformLocation(mProg.initSearchRadiusGrid, "stretchFactor"), 1.0f / cellSideLength, 1.0f / cellSideLength);
		glUniform1f(glGetUniformLocation(mProg.initSearchRadiusGrid, "pixelGridCellSideLength"), cellSideLength);
		glUniform1f(glGetUniformLocation(mProg.initSearchRadiusGrid, "pixelToWorldFactor"), mViewPlaneRT.x / (0.5f * mWidth * mZNear));
		glUniform1i(glGetUniformLocation(mProg.initSearchRadiusGrid, "K"), params.K);

		bindUniformSampler(mProg.initSearchRadiusGrid, 0, "texCounterGrid", mTex.counterGrid);
		bindUniformSampler(mProg.initSearchRadiusGrid, 1, "texPointPos", mTex.pointPos);


		drawScreenPoints();

		// reset states
		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}



	// This shit is faster
	void AutoSplats::adaptSearchRadiiInstantGather(int iter, float zNear, float zFar, const Params& params)
	{
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);


		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
		setDrawBuffer(1);				// rasterize only into the sr tex without clearing it

		glUseProgram(mProg.rangeSearchInstantGather);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchInstantGather, "nearFar"), zNear, zFar);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchInstantGather, "invViewPlaneRT"), mInvViewPlaneRT.x, mInvViewPlaneRT.y);
		glUniform1f(glGetUniformLocation(mProg.rangeSearchInstantGather, "vPixelDiagonal"), mVPixelDiagonal);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchInstantGather, "invVpSize"), mInvVpSize.x, mInvVpSize.y);
		// uniforms qsSplatBias = 0, qsUseSafeQuad = false, qsUseSafeDepth = false

		glUniform2i(glGetUniformLocation(mProg.rangeSearchInstantGather, "vpSize"), mWidth, mHeight);
		glUniform1i(glGetUniformLocation(mProg.rangeSearchInstantGather, "K"), params.K);
		glUniform1i(glGetUniformLocation(mProg.rangeSearchInstantGather, "iteration"), iter);
		glUniformMatrix4fv(glGetUniformLocation(mProg.rangeSearchInstantGather, "viewMatrix"), 1, true, mModelViewMatrix);
		//glUniform1i(glGetUniformLocation(mProg.rangeSearchInstantGather, "maxZeroIterations"), ... );
		//glUniform1f(glGetUniformLocation(mProg.rangeSearchInstantGather, "breakDensityFactor"), ... );

		bindUniformSampler(mProg.rangeSearchInstantGather, 0, "wPosTex", mTex.pointPos);
		bindUniformSampler(mProg.rangeSearchInstantGather, 1, "searchRadiiTex1", mTex.sr0);
		bindUniformImage(mProg.rangeSearchInstantGather, 0, "imgNeighCount", mTex.neighCount, GL_READ_WRITE, GL_R32UI);

		drawScreenPoints();

		// Here we set a memory barrier in order to ensure that all reads to images are visible by the next iteration
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT);		// second bit for tex inspector

		//-----------------------------------------------------------------------------
		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);	glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}



	void AutoSplats::adaptSearchRadii(int iter, float zNear, float zFar, const Params& params)
	{
		startTimer();

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);


		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);


		// DISTRIBUTE PASS
		//-----------------------------------------------------------------------------
		setDrawBuffer(3);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(mProg.rangeSearchDistribute);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchDistribute, "nearFar"), zNear, zFar);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchDistribute, "invViewPlaneRT"), mInvViewPlaneRT.x, mInvViewPlaneRT.y);
		glUniform1f(glGetUniformLocation(mProg.rangeSearchDistribute, "vPixelDiagonal"), mVPixelDiagonal);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchDistribute, "invVpSize"), mInvVpSize.x, mInvVpSize.y);
		// uniforms qsSplatBias = 0, qsUseSafeQuad = false, qsUseSafeDepth = false

		glUniform2i(glGetUniformLocation(mProg.rangeSearchDistribute, "vpSize"), mWidth, mHeight);
		glUniform1i(glGetUniformLocation(mProg.rangeSearchDistribute, "K"), params.K);
		glUniform1i(glGetUniformLocation(mProg.rangeSearchDistribute, "iteration"), iter);
		glUniformMatrix4fv(glGetUniformLocation(mProg.rangeSearchDistribute, "viewMatrix"), 1, true, mModelViewMatrix);
		//glUniform1i(glGetUniformLocation(mProg.rangeSearchDistribute, "maxZeroIterations"), ... );
		//glUniform1f(glGetUniformLocation(mProg.rangeSearchDistribute, "breakDensityFactor"), ... );

		bindUniformSampler(mProg.rangeSearchDistribute, 0, "wPosTex", mTex.pointPos);
		bindUniformSampler(mProg.rangeSearchDistribute, 1, "searchRadiiTex1", mTex.sr0);
		bindUniformSampler(mProg.rangeSearchDistribute, 2, "searchRadiiTex2", mTex.sr1);

		drawScreenPoints();


		timings.rsDist += stopTimer();
		startTimer();


		// GATHER PASS
		//-----------------------------------------------------------------------------
		setDrawBuffers(1, 2);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(mProg.rangeSearchGather);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchGather, "nearFar"), zNear, zFar);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchGather, "invViewPlaneRT"), mInvViewPlaneRT.x, mInvViewPlaneRT.y);
		glUniform1f(glGetUniformLocation(mProg.rangeSearchGather, "vPixelDiagonal"), mVPixelDiagonal);
		glUniform2f(glGetUniformLocation(mProg.rangeSearchGather, "invVpSize"), mInvVpSize.x, mInvVpSize.y);
		// uniforms qsSplatBias = 0, qsUseSafeQuad = false, qsUseSafeDepth = false

		glUniform2i(glGetUniformLocation(mProg.rangeSearchGather, "vpSize"), mWidth, mHeight);
		glUniformMatrix4fv(glGetUniformLocation(mProg.rangeSearchGather, "viewMatrix"), 1, true, mModelViewMatrix);
		//glUniform1i(glGetUniformLocation(mProg.rangeSearchDistribute, "maxZeroIterations"), ... );
		//glUniform1f(glGetUniformLocation(mProg.rangeSearchDistribute, "breakDensityFactor"), ... );

		bindUniformSampler(mProg.rangeSearchGather, 0, "wPosTex", mTex.pointPos);
		bindUniformSampler(mProg.rangeSearchGather, 1, "feedbackSearchRadiiTex", mTex.fr);

		// Draw Points
		drawScreenPoints();


		//-----------------------------------------------------------------------------
		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);	glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_BLEND);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		timings.rsGather += stopTimer();
	}




	void AutoSplats::finalDistribute(float zNear, float zFar)
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendEquation(GL_MAX);

    glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
		setDrawBuffer(4);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(mProg.finalDistribute);
		glUniform2f(glGetUniformLocation(mProg.finalDistribute, "nearFar"), zNear, zFar);
		glUniform2f(glGetUniformLocation(mProg.finalDistribute, "invViewPlaneRT"), mInvViewPlaneRT.x, mInvViewPlaneRT.y);
		glUniform1f(glGetUniformLocation(mProg.finalDistribute, "vPixelDiagonal"), mVPixelDiagonal);
		glUniform2f(glGetUniformLocation(mProg.finalDistribute, "invVpSize"), mInvVpSize.x, mInvVpSize.y);
		// uniforms qsSplatBias = 0, qsUseSafeQuad = false, qsUseSafeDepth = false
		glUniformMatrix4fv(glGetUniformLocation(mProg.finalDistribute, "viewMatrix"), 1, true, mModelViewMatrix);

		bindUniformSampler(mProg.finalDistribute, 0, "wPosTex", mTex.pointPos);
		bindUniformSampler(mProg.finalDistribute, 1, "searchRadiiTex1", mTex.sr0);

    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
		drawScreenPoints();

		//-----------------------------------------------------------------------------
		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);	glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_BLEND);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}



	void AutoSplats::gatherHCov(float zNear, float zFar)
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendEquation(GL_FUNC_ADD);

		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenAS);
		setDrawBuffers(5, 6, 7);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(mProg.gatherHCov);
		glUniform2f(glGetUniformLocation(mProg.gatherHCov, "nearFar"), zNear, zFar);
		glUniform2f(glGetUniformLocation(mProg.gatherHCov, "invViewPlaneRT"), mInvViewPlaneRT.x, mInvViewPlaneRT.y);
		glUniform1f(glGetUniformLocation(mProg.gatherHCov, "vPixelDiagonal"), mVPixelDiagonal);
		glUniform2f(glGetUniformLocation(mProg.gatherHCov, "invVpSize"), mInvVpSize.x, mInvVpSize.y);
		// uniforms qsSplatBias = 0, qsUseSafeQuad = false, qsUseSafeDepth = false
		glUniformMatrix4fv(glGetUniformLocation(mProg.gatherHCov, "viewMatrix"), 1, true, mModelViewMatrix);

		bindUniformSampler(mProg.gatherHCov, 0, "wPosTex", mTex.pointPos);
		bindUniformSampler(mProg.gatherHCov, 1, "searchRadiiTex1", mTex.sr0);
		bindUniformSampler(mProg.gatherHCov, 2, "finalFeedbackRadiiTex", mTex.frFinal);

		// Draw Points
		drawScreenPoints();


		//-----------------------------------------------------------------------------
		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);	glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glDisable(GL_BLEND);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}



	void AutoSplats::computeSplatVBOs(const Params& params)
	{
		glDisable(GL_DEPTH_TEST);

		glUseProgram(mProg.computeSplatVBOs);
		bindUniformSampler(mProg.computeSplatVBOs, 0, "searchRadiiTex", mTex.sr0);
		bindUniformSampler(mProg.computeSplatVBOs, 1, "wPosTex", mTex.pointPos);
		bindUniformSampler(mProg.computeSplatVBOs, 2, "texHCovMean", mTex.hcovMean);
		bindUniformSampler(mProg.computeSplatVBOs, 3, "texHCov1", mTex.hcov1);
		bindUniformSampler(mProg.computeSplatVBOs, 4, "texHCov2", mTex.hcov2);
    bindUniformSampler(mProg.computeSplatVBOs, 5, "correctNormalTex", mTex.correctNormals);
		glUniform1i(glGetUniformLocation(mProg.computeSplatVBOs, "projectToMean"), params.projectToMean);


		// Transform Feedback Initialization
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, mASPositionVbo);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, mASNormalsVbo);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 2, mASSplatAxesVbo);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 3, mASCorrectNormalsVbo);
		glEnable(GL_RASTERIZER_DISCARD);
		glBeginTransformFeedback(GL_POINTS);

		drawScreenPoints();

		glEndTransformFeedback();
		glDisable(GL_RASTERIZER_DISCARD);


		// reset states
		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, 0);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 1, 0);
		glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 2, 0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 3, 0);
	}

	void AutoSplats::surfaceSplatting(const mat4& modelViewMatrix, const mat4& projMatrix, float zNear, float zFar, const Params& params)
	{
		glViewport(0, 0, mWidth, mHeight);
		glClearColor(0, 0, 0, 0);
		glClearDepth(1.0f);

		// bind surface splatting screen FBO
		glBindFramebuffer(GL_FRAMEBUFFER, mFboScreenSS);
//    glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glEnableVertexAttribArray(0);	glBindBuffer(GL_ARRAY_BUFFER, mASPositionVbo);	glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, NULL);
    glEnableVertexAttribArray(1);	glBindBuffer(GL_ARRAY_BUFFER, mASNormalsVbo);	glVertexAttribPointer(1, 3, GL_FLOAT, false, 0, NULL);
    glEnableVertexAttribArray(2);	glBindBuffer(GL_ARRAY_BUFFER, mASSplatAxesVbo);	glVertexAttribPointer(2, 4, GL_FLOAT, false, 0, NULL);
    glEnableVertexAttribArray(3);	glBindBuffer(GL_ARRAY_BUFFER, mASCorrectNormalsVbo);	glVertexAttribPointer(3, 3, GL_FLOAT, false, 0, NULL);


		// 1. Depth Pass
		//-----------------------------------------------------------------------------
		setDrawBuffer(0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		glUseProgram(mProg.splatDepth);
		glUniformMatrix4fv(glGetUniformLocation(mProg.splatDepth, "mvMatrix"), 1, true, modelViewMatrix);
		glUniformMatrix4fv(glGetUniformLocation(mProg.splatDepth, "normalMatrix"), 1, true, modelViewMatrix);
		glUniformMatrix4fv(glGetUniformLocation(mProg.splatDepth, "projMatrix"), 1, true, projMatrix);
		glUniform1f(glGetUniformLocation(mProg.splatDepth, "radiusFac"), params.splatRadiusFactor);
		glDrawArrays(GL_POINTS, 0, mNumScreenPoints);
//return;
		
//glDrawBuffer(GL_COLOR_ATTACHMENT0);
//glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// 2. Accumulation Pass
		//-----------------------------------------------------------------------------
		setDrawBuffers(1, 2);
		glClear(GL_COLOR_BUFFER_BIT);
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE);

		glUseProgram(mProg.splatAccum);
		bindUniformSampler(mProg.splatAccum, 0, "texDepth", mTex.depth);

		glUniformMatrix4fv(glGetUniformLocation(mProg.splatAccum, "mvMatrix"), 1, true, modelViewMatrix);
		glUniformMatrix4fv(glGetUniformLocation(mProg.splatAccum, "normalMatrix"), 1, true, modelViewMatrix);
		glUniformMatrix4fv(glGetUniformLocation(mProg.splatAccum, "projMatrix"), 1, true, projMatrix);
		glUniform1f(glGetUniformLocation(mProg.splatAccum, "radiusFac"), params.splatRadiusFactor);
		glDrawArrays(GL_POINTS, 0, mNumScreenPoints);
		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);
//    return;

		// 3. Final rendering pass
		//-----------------------------------------------------------------------------
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glEnable(GL_DEPTH_TEST);
		glDisable(GL_BLEND);

		glUseProgram(mProg.splatFinal);
		bindUniformSampler(mProg.splatFinal, 0, "texAccumNormalDepth", mTex.accumNormalDepth);
		bindUniformSampler(mProg.splatFinal, 1, "texAccumColorWeight", mTex.accumColorWeight);

		glUniformMatrix4fv(glGetUniformLocation(mProg.splatFinal, "projMatrixI"), 1, false, inverse(projMatrix));
		glUniformMatrix4fv(glGetUniformLocation(mProg.splatFinal, "mvMatrixI"), 1, false, inverse(modelViewMatrix));
		glUniform1f(glGetUniformLocation(mProg.splatFinal, "near"), zNear);
		glUniform1f(glGetUniformLocation(mProg.splatFinal, "far"), zFar);

		// use shader default temporarily
		//	glUniform3fv(glGetUniformLocation(mProgSplatFinal, "kads"), 3, &props.kads.x );
		//	glUniform1f(glGetUniformLocation(mProgSplatFinal, "shininess"), props.shininess );


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableVertexAttribArray(0);

		//-----------------------------------------------------------------------------
		glUseProgram(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glActiveTexture(GL_TEXTURE0);		glBindTexture(GL_TEXTURE_2D, 0);
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		//glFinish
	}



	/// flushes the input point vbo through the pipeline
	void AutoSplats::drawPoints(uint vbo, int numComponents, uint glType, uint numPoints)
	{
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, numComponents, glType, false, 0, 0);
		glDrawArrays(GL_POINTS, 0, numPoints);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableVertexAttribArray(0);
	}


	void AutoSplats::drawScreenPoints()
	{
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, mVboScreenPoints);
		glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, 0);
		glDrawArrays(GL_POINTS, 0, mNumScreenPoints);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glDisableVertexAttribArray(0);
	}
}

