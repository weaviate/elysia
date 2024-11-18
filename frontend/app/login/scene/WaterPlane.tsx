"use client";

import React, { useEffect, useState } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

function WaterPlane() {
  const { scene } = useThree();
  const [material, setMaterial] = useState<THREE.Material | null>(null);

  useFrame((state) => {
    if (material) {
      /* eslint-disable @typescript-eslint/no-explicit-any */
      (material as any).uTime.value = state.clock.getElapsedTime();
    }
  });

  useEffect(() => {
    if (!scene || !scene.fog) return;

    const baseMaterial = new THREE.MeshStandardMaterial({
      color: "#A436AE",
      normalMap: new THREE.TextureLoader().load(
        "/waternormals.jpg",
        (texture) => {
          texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
          texture.repeat.set(5, 5); // Increase this number to make the texture repeat more
        }
      ),
      normalScale: new THREE.Vector2(2, 2),
      normalMapType: THREE.TangentSpaceNormalMap,
      transparent: true,
      roughness: 1,
      metalness: 0,
      opacity: 0.95,
    });
    /* eslint-disable @typescript-eslint/no-explicit-any */
    (baseMaterial as any).uTime = { value: 0 };

    baseMaterial.onBeforeCompile = (shader) => {
      /* eslint-disable @typescript-eslint/no-explicit-any */
      shader.uniforms.uTime = (baseMaterial as any).uTime;

      shader.vertexShader = shader.vertexShader.replace(
        "#include <common>",
        `
        #include <common>
        uniform float uTime;
        varying vec2 vUv;
        varying vec3 vPos;
        varying vec3 vViewDir;
        `
      );

      shader.vertexShader = shader.vertexShader.replace(
        "#include <begin_vertex>",
        `
        #include <begin_vertex>
        vUv = uv;
        vec3 pos = transformed;
  
        float waveStrength = 0.1;
        float waveSpeed = 0.5;
  
        // Create more complex wave patterns with varied frequencies and phases
        float wave1 = sin(pos.x * 2.0 + uTime) * cos(pos.y + uTime * 0.8);
        float wave2 = sin(pos.y * 3.0 + uTime * waveSpeed) * sin(pos.x * 1.5 - uTime * 0.3);
        float wave3 = sin((pos.x + pos.y) * 5.0 + uTime * waveSpeed);
        float wave4 = cos(pos.x * 4.0 - pos.y * 2.0 + uTime * 0.5) * 0.5;
  
        // Combine waves with different weights
        pos.z += (wave1 * 0.4 + wave2 * 0.3 + wave3 * 0.2 + wave4 * 0.1) * waveStrength;

                float dx = waveStrength * (
          0.8 * cos(pos.x * 2.0 + uTime) * cos(pos.y + uTime * 0.8) +
          0.45 * sin(pos.x * 1.5 - uTime * 0.3) +
          1.0 * cos((pos.x + pos.y) * 5.0 + uTime * waveSpeed) +
          0.2 * -sin(pos.x * 4.0 - pos.y * 2.0 + uTime * 0.5) * 4.0
        );
        
        float dy = waveStrength * (
          0.4 * sin(pos.x * 2.0 + uTime) * -sin(pos.y + uTime * 0.8) +
          0.9 * cos(pos.y * 3.0 + uTime * waveSpeed) +
          1.0 * cos((pos.x + pos.y) * 5.0 + uTime * waveSpeed) +
          0.2 * sin(pos.x * 4.0 - pos.y * 2.0 + uTime * 0.5) * 2.0
        );

        // Construct the new normal
        vec3 newNormal = normalize(vec3(-dx, -dy, 1.0));
  
        transformed = pos;
        vPos = pos;
        `
      );
    };

    setMaterial(baseMaterial);
  }, [scene]);

  if (!material) return null;

  return (
    <mesh
      rotation={[-Math.PI / 2, 0, 0]}
      receiveShadow
      castShadow
      material={material}
    >
      <planeGeometry args={[20, 20, 100, 100]} />
    </mesh>
  );
}

export default WaterPlane;
