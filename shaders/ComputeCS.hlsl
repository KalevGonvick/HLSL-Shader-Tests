#define M_PI 3.1415926535897932384626
#define g 9.81
#define TRANSFORM_INVERSE 1
#define BUTTERFLY_COUNT 16
#define LENGTH 256
#define COLPASS

//pre-calculated h0k and k0minuk values
Texture2D<float4> tilde_h0k_val : register(t0);
Texture2D<float4> tilde_h0minusk_val : register(t1);

RWTexture2D<float3> tilde_hkt_real : register(u0);
RWTexture2D<float3> tilde_hkt_im : register(u1);
RWTexture2D<float3> TextureTargetR  : register(u2);
RWTexture2D<float3> TextureTargetI : register(u3);

uniform int N;
uniform int L;
uniform float A;
uniform float2 windDirection;
uniform float windSpeed;
uniform float time;
struct complex
{
	float real;
	float im;
};

complex cmul(complex c0, complex c1) 
{
	complex c;
	c.real = c0.real * c1.real - c0.im * c1.im;
	c.im = c0.real * c1.im + c0.im * c1.real;
	return c;
}

complex cadd(complex c0, complex c1)
{
	complex c;
	c.real = c0.real + c1.real;
	c.im = c0.im + c1.im;
	return c;
}

complex cconj(complex c)
{
	complex c_conj = complex(c.real, -c.im);
	return c_conj;
}


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

void mainButterfly( uint3 dispatchThreadId, uint groupIndex) {
#ifdef ROWPASS
	uint2 texCoord = dispatchThreadId.xy;
	if(fmod(groupIndex, 16) == 0 && groupIndex != 0) {
		uint new_y = texCoord.y * groupIndex/16;
		texCoord.y = new_y;
	}
	
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

[numthreads(16, 16, 1)]
void main(
	uint3 groupId : SV_GroupID,
	uint3 groupThreadId : SV_GroupThreadID,
	uint3 dispatchThreadId : SV_DispatchThreadID,
	uint groupIndex : SV_GroupIndex)
{
	
	float2 x = int2(dispatchThreadId.xy) - float(N)/2.0;
	float2 k = float2(2.0 * M_PI * x.x / L, 2.0 * M_PI * x.y / L);
	float mag = length(k);
	if (mag < 0.00001) mag = 0.00001;
	float magSq = mag * mag;
	float w = sqrt(g*mag);
	
	float2 tilde_h0k_values = float2(tilde_h0k_val[dispatchThreadId.xy].xy);
	complex fourier_cmp = complex(tilde_h0k_values.x, tilde_h0k_values.y);
	
	float2 tilde_h0minusk_values = float2(tilde_h0minusk_val[dispatchThreadId.xy].xy);
	complex fourier_cmp_conj = cconj(complex(tilde_h0minusk_values.x, tilde_h0minusk_values.y));
	
	float cos_w_t = cos(w*time);
	float sin_w_t = sin(w*time);
	
	// euler
	complex exp_iwt = complex(cos_w_t, sin_w_t);
	complex exp_iwt_inv = complex(cos_w_t, -sin_w_t);
	
	//dy
	complex h_k_t_dy = cadd(cmul(fourier_cmp, exp_iwt), cmul(fourier_cmp_conj, exp_iwt_inv));
	
	//dx
	complex dx = complex(0.0,-k.x/mag);
	complex h_k_t_dx = cmul(dx, h_k_t_dy);
	
	//dz
	complex dy = complex(0.0, -k.y/mag);
	complex h_k_t_dz = cmul(dy, h_k_t_dy);
	tilde_hkt_real[dispatchThreadId.xy] = float4(h_k_t_dx.real, h_k_t_dy.real, h_k_t_dz.real, 1);
	tilde_hkt_im[dispatchThreadId.xy] = float4(h_k_t_dx.im, h_k_t_dy.im, h_k_t_dz.im, 1);
	GroupMemoryBarrierWithGroupSync();
	mainButterfly(dispatchThreadId, groupIndex);
}






