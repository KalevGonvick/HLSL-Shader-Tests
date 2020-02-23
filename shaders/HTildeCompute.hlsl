#define M_PI 3.141592653589793238
#define g 9.81


RWTexture2D<float4> tilde_h0k_data : register(u0);
RWTexture2D<float4> tilde_h0minusk_data : register(u1);

Texture2D<float4> noise_r0 : register(t0);
Texture2D<float4> noise_i0 : register(t1);
Texture2D<float4> noise_r1 : register(t2);
Texture2D<float4> noise_i1 : register(t3);
SamplerState noise_r0_sampler;
SamplerState noise_i0_sampler;
SamplerState noise_r1_sampler;
SamplerState noise_i1_sampler;
int N = 256;
int L = 1000;
float A = 4;
float2 windDirection = float2(1.0, 1.0);
float windSpeed = 40.0;

float4 randomGauss(float2 gId) 
{
	float2 texCoord = float2(gId.xy) / float(N);
	//uint index
	//uint w1,h1;
	
	float noise00 = clamp(noise_r0.Sample(noise_r0_sampler, texCoord).x, 0.001, 1.0);
	float noise01 = clamp(noise_i0.Sample(noise_i0_sampler, texCoord).x, 0.001, 1.0);
	float noise02 = clamp(noise_r1.Sample(noise_r1_sampler, texCoord).x, 0.001, 1.0);
	float noise03 = clamp(noise_i1.Sample(noise_i1_sampler, texCoord).x, 0.001, 1.0);
	
	float u0 = 2.0*M_PI*noise00;
	float v0 = sqrt(-2 * log(noise01));
	float u1 = 2.0*M_PI*noise02;
	float v1 = sqrt(-2.0*log(noise03));

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
	
	float2 x = float2(dispatchThreadId.xy) - float(N);
	float2 k = float2(2.0 * M_PI * x.x / L, 2.0*M_PI*x.y / L);
	float L_ = (windSpeed * windSpeed) / g;
	float mag = length(k);
	if (mag < 0.00001) mag = 0.00001;
	float magSq = mag * mag;

	float h0_k = clamp(sqrt((A / (magSq*magSq))
		*pow(dot(normalize(k), normalize(windDirection)), 2)
		*exp(-(1.0 / (magSq*L_*L_)))
		*exp(-magSq * pow(L / 2000.0, 2.0))) / sqrt(2.0), -4000, 4000);

	float h0_minusk = clamp(sqrt((A / (magSq*magSq))
		*pow(dot(normalize(-k), normalize(windDirection)), 2)
		*exp(-(1.0 / (magSq*L_*L_)))
		*exp(-magSq * pow(L / 2000.0, 2.0))) / sqrt(2.0), -4000, 4000);

	float4 gaussRnd = randomGauss(float2(dispatchThreadId.x, dispatchThreadId.y));
	float4 h_t_mk_val = float4(gaussRnd.z * h0_minusk,gaussRnd.w * h0_minusk, 0, 1);
	float4 h_t_k_val = float4(gaussRnd.x * h0_k,gaussRnd.y * h0_k, 0, 1);
	tilde_h0k_data[dispatchThreadId.xy] = h_t_k_val;
	tilde_h0minusk_data[dispatchThreadId.xy] = h_t_mk_val;
}




