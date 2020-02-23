#define M_PI 3.1415926535897932384626
#define g 9.81

#define TRANSFORM_INVERSE 1
#define BUTTERFLY_COUNT 16
#define LENGTH 256
#define COLPASS

//pre-calculated h0k and k0minuk values
Texture2D<float4> tilde_hkt_real : register(t0);
Texture2D<float4> tilde_hkt_im : register(t1);

RWTexture2D<float3> TextureTargetR  : register(u2);
RWTexture2D<float3> TextureTargetI  : register(u3);

void GetButterflyValues(uint passIndex, uint x, out uint2 indices, out float2 weights)
{
	int sectionWidth = 2 << passIndex;
	int halfSectionWidth = sectionWidth / 2;

	int sectionStartOffset = x & ~(sectionWidth - 1);
	int halfSectionOffset = x & (halfSectionWidth - 1);
	int sectionOffset = x & (sectionWidth - 1);

	sincos( 2.0*M_PI*sectionOffset / (float)sectionWidth, weights.y, weights.x );
	weights.y = -weights.y;

	indices.x = sectionStartOffset + halfSectionOffset;
	indices.y = sectionStartOffset + halfSectionOffset + halfSectionWidth;

	if (passIndex == 0)
	{
		indices = reversebits(indices) >> (32 - BUTTERFLY_COUNT) & (LENGTH - 1);
	}
}

groupshared float3 pingPongArray[4][LENGTH];
void ButterflyPass(int passIndex, uint x, uint t0, uint t1, out float3 resultR, out float3 resultI)
{
	uint2 Indices;
	float2 Weights;
	GetButterflyValues(passIndex, x, Indices, Weights);

	float3 inputR1 = pingPongArray[t0][Indices.x];
	float3 inputI1 = pingPongArray[t1][Indices.x];

	float3 inputR2 = pingPongArray[t0][Indices.y];
	float3 inputI2 = pingPongArray[t1][Indices.y];
	resultR = (inputR1 + Weights.x * inputR2 + Weights.y * inputI2) * 0.5;
	resultI = (inputI1 - Weights.y * inputR2 + Weights.x * inputI2) * 0.5;

}

void ButterflyPassFinalNoI(int passIndex, int x, int t0, int t1, out float3 resultR)
{
	uint2 Indices;
	float2 Weights;
	GetButterflyValues(passIndex, x, Indices, Weights);

	float3 inputR1 = pingPongArray[t0][Indices.x];

	float3 inputR2 = pingPongArray[t0][Indices.y];
	float3 inputI2 = pingPongArray[t1][Indices.y];

	resultR = (inputR1 + Weights.x * inputR2 + Weights.y * inputI2) * 0.5;
}

[numthreads(1, 1, 1)]
void mainButterfly(
uint3 dispatchThreadId : SV_DispatchThreadID
) {
	#ifdef ROWPASS
	uint2 texturePos = uint2( dispatchThreadId.xy );
#else
	uint2 texturePos = uint2( dispatchThreadId.yx );
#endif

	// Load entire row or column into scratch array
	pingPongArray[0][dispatchThreadId.x].xyz = tilde_hkt_real[texturePos];

	pingPongArray[1][dispatchThreadId.x].xyz = tilde_hkt_im[texturePos];

	
	uint4 textureIndices = uint4(0, 1, 2, 3);

	
	for (int i = 0; i < BUTTERFLY_COUNT-1; i++)
	{
		GroupMemoryBarrierWithGroupSync();
		ButterflyPass(i,dispatchThreadId.x,textureIndices.x, textureIndices.y,pingPongArray[textureIndices.z][dispatchThreadId.x].xyz,pingPongArray[textureIndices.w][dispatchThreadId.x].xyz);
		textureIndices.xyzw = textureIndices.zwxy;
	}

	// Final butterfly will write directly to the target texture
	GroupMemoryBarrierWithGroupSync();

	// The final pass writes to the output UAV texture
#if defined(COLPASS) && defined(TRANSFORM_INVERSE)
	// last pass of the inverse transform. The imaginary value is no longer needed
	ButterflyPassFinalNoI(BUTTERFLY_COUNT - 1, dispatchThreadId.x, textureIndices.x, textureIndices.y, TextureTargetR[texturePos]);
#else
	ButterflyPass(BUTTERFLY_COUNT - 1, dispatchThreadId.x, textureIndices.x, textureIndices.y, TextureTargetR[texturePos], TextureTargetI[texturePos]);
#endif
}



