uniform vec3 fogColor;
uniform float fogNear;
uniform float fogFar;
varying vec2 vUv;
varying vec3 vPos;
varying vec4 shadowDebugColor;


void main() {
  // Compute normals using derivatives
  vec3 dx = dFdx(vPos);
  vec3 dy = dFdy(vPos);
  vec3 normal = normalize(cross(dx, dy));

  // Simple directional light
  vec3 lightDir = normalize(vec3(0.0, 1.0, 0.5));
  float diff = max(dot(normal, lightDir), 0.0);

  // Specular highlights
  vec3 viewDir = normalize(cameraPosition - vPos);
  vec3 reflectDir = reflect(-lightDir, normal);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);

  // Combine diffuse and specular components
  vec3 baseColor = vec3(0.0, 0.4, 0.8);
  vec3 color = baseColor * diff + vec3(1.0) * spec * 0.5;

  // Calculate fog - modified to be more linear
  float depth = length(cameraPosition - vPos);
  float fogFactor = clamp((depth - fogNear) / (fogFar - fogNear), 0.0, 1.0);

  // Mix the color with fog
  color = mix(color, fogColor, fogFactor);

  gl_FragColor = vec4(color, 1);


  // To visualize the shadow coordinates:
  //gl_FragColor = shadowDebugColor * 0.5 + 0.5; // Remap from [-1,1] to [0,1]
}
