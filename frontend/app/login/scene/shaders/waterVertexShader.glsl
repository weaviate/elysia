uniform float uTime;
varying vec2 vUv;
varying vec3 vPos;
uniform mat4 shadowMatrix;
varying vec4 vShadowCoord;
varying vec4 shadowDebugColor;
uniform mat4 directionalShadowMatrix;
uniform mat4 spotShadowMatrix;
uniform mat4 pointShadowMatrix;
varying vec4 vDirectionalShadowCoord;

void main() {
  vUv = uv;
  vec3 pos = position;

  float waveStrength = 0.15;
  float waveSpeed = 0.25;

  // Create more complex wave patterns with varied frequencies and phases
  float wave1 = sin(pos.x * 2.0 + uTime) * cos(pos.y + uTime * 0.8);
  float wave2 = sin(pos.y * 3.0 + uTime * waveSpeed) * sin(pos.x * 1.5 - uTime * 0.3);
  float wave3 = sin((pos.x + pos.y) * 5.0 + uTime * waveSpeed);
  float wave4 = cos(pos.x * 4.0 - pos.y * 2.0 + uTime * 0.5) * 0.5;
  
  // Combine waves with different weights
  pos.z += (wave1 * 0.4 + wave2 * 0.3 + wave3 * 0.2 + wave4 * 0.1) * waveStrength;

  vShadowCoord = shadowMatrix * vec4(pos, 1.0);
  shadowDebugColor = vShadowCoord;

  vDirectionalShadowCoord = directionalShadowMatrix * vec4(pos, 1.0);

  vPos = pos;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
