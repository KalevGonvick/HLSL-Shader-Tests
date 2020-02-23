#define M_PI 3.1415926535897932384626
#define g 9.81

// 
RWTexture2D<float4> tilde_h0k_value : register(u0);
RWTexture2D<float4> tilde_h0minusk_value : register(u1);

// hkty, hktx, and hktz results (part1)
RWTexture2D<float4> tilde_hkt_dz : register(u2);
RWTexture2D<float4> tilde_hkt_dy : register(u3);
RWTexture2D<float4> tilde_hkt_dx : register(u4);

// noise samples - precomputed as png
Texture2D<float4> noise_r0 : register(t0);
Texture2D<float4> noise_i0 : register(t1);
Texture2D<float4> noise_r1 : register(t2);
Texture2D<float4> noise_i1 : register(t3);

uniform int N;
uniform int L;
uniform float A;
uniform float2 windDirection;
uniform float windSpeed;
uniform float time;

SamplerState noise_r0_sampler
{
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState noise_i0_sampler
{
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState noise_r1_sampler
{
    AddressU = Wrap;
    AddressV = Wrap;
};

SamplerState noise_i1_sampler
{
    AddressU = Wrap;
    AddressV = Wrap;
};

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

float4 randomGauss(float2 gId) 
{
	// position --> texel
	float2 texCoord = float2(gId.xy) / float(N);
	
	float noise00 = saturate(noise_r0.GatherRed(noise_r0_sampler, texCoord));
	float noise01 = saturate(noise_i0.GatherRed(noise_i0_sampler, texCoord));
	float noise02 = saturate(noise_r1.GatherRed(noise_r1_sampler, texCoord));
	float noise03 = saturate(noise_i1.GatherRed(noise_i1_sampler, texCoord));
	
	float u0 = 2.0*M_PI*noise00;
	float v0 = sqrt(-2 * log(noise01));
	float u1 = 2.0*M_PI*noise02;
	float v1 = sqrt(-2.0 * log(noise03));

	float4 rndFinal = float4(v0 * cos(u0), v0 * sin(u0), v1 * cos(u1), v1 * sin(u1));
	return rndFinal;
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
	float L_ = (windSpeed * windSpeed) / g;
	float mag = length(k);
	if (mag < 0.00001) mag = 0.00001;
	float magSq = mag * mag;

	float h0_k = clamp(sqrt((A / (magSq*magSq))
		*pow(dot(normalize(k), normalize(windDirection)), 6.0)
		*exp(-(1.0 / (magSq*L_*L_)))
		*exp(-magSq * pow(L / 2000.0, 2.0))) / sqrt(2.0), -4000, 4000);

	float h0_minusk = clamp(sqrt((A / (magSq*magSq))
		*pow(dot(normalize((k * -1.0)), normalize(windDirection)), 6.0)
		*exp(-(1.0 / (magSq*L_*L_)))
		*exp(-magSq * pow(L / 2000.0, 2.0))) / sqrt(2.0), -4000, 4000);
		
	float4 gaussRnd = randomGauss(float2(dispatchThreadId.x, dispatchThreadId.y));
	float4 h_t_mk_val = float4(gaussRnd.zw * h0_minusk, 0, 1);
	float4 h_t_k_val = float4(gaussRnd.xy * h0_k, 0, 1);
	
	
	tilde_h0k_value[dispatchThreadId.xy] = h_t_k_val;
	tilde_h0minusk_value[dispatchThreadId.xy] = h_t_mk_val;
	
	
	float w = sqrt(g*mag);
	float2 tilde_h0k_values_rg = tilde_h0k_value[dispatchThreadId.xy].xy;
	complex fourier_cmp = complex(tilde_h0k_values_rg.x, tilde_h0k_values_rg.y);
	float2 tilde_h0minusk_values_rg = tilde_h0minusk_value[dispatchThreadId.xy].xy;
	complex fourier_cmp_conj = cconj(complex(tilde_h0minusk_values_rg.x, tilde_h0minusk_values_rg.y));
	
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

	GroupMemoryBarrierWithGroupSync();
	
	tilde_hkt_dy[dispatchThreadId.xy] = float4(h_k_t_dy.real, h_k_t_dy.im, 0, 1);
	tilde_hkt_dx[dispatchThreadId.xy] = float4(h_k_t_dx.real, h_k_t_dx.im, 0, 1);
	tilde_hkt_dz[dispatchThreadId.xy] = float4(h_k_t_dz.real, h_k_t_dz.im, 0, 1);
	
	GroupMemoryBarrierWithGroupSync();
}

