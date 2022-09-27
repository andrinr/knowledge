# GLSL

## I/O

### Vertex / Fragment Shader communication

````glsl

varying <type> <variable>;

varying vec2 vector;

````

### Uniforms

## Raytracer

### Most basic sphere raytracer:

````glsl
float map(vec3 p){
    vec3 q = fract(p) * 2.0 - 1.0;
    return length(q) - 0.25;
}

float trace(vec3 origin, vec3 ray){
    float t = 0.;
    
    for (int i = 0; i < 32; ++i){
        vec3 p = origin + ray * t;
        float d = map(p);
        t += d* 0.5;
    }
    return t;
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.xy;
    
    uv = uv * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;
    c
    vec3 ray = normalize(vec3(uv, 1.0));
	vec3 origin = vec3(0.0, 0.0, iTime);
    
    float t = trace(origin, ray);
    float fog = 1.0 / (1.0 + t*t* 0.1);
    
    vec3 fc = vec3(fog);

    // Output to screen
    fragColor = vec4(fc,1.0);
} 
````

