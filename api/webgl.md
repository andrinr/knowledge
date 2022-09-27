# WebGL

Warning: WebGL 2.0 is not supported in all browsers, which sucks. Because WebGL is bad. But what can we do?

Wait for WebGPU?

## Extensions

The namings from the Mozilla documentation may sometimes not be correct. Print the available extensions like this for certainty:

```typescript
var available_extensions = gl.getSupportedExtensions();
```

To enable an extension:

```typescript
var ext = gl.getExtension('WEBGL_color_buffer_float');
```

If the method returns a null pointer this means that the extension is NOT available.

## Float textures in WebGL 2.0

In WebGL 2.0 float texture are enabled by default. However the extension OES_texture_float_linear still has to enabled in order to use float textures with linear filtering. This extension strangely doesn’t need to be enabled with WebGL 1.0.

