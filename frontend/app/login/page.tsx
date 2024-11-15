// page.tsx or LoginPage.tsx
"use client";

import React, { useRef } from "react";
import { useRouter } from "next/navigation";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { useEffect } from "react";
import { Environment } from "@react-three/drei";

import WaterPlane from "./scene/WaterPlane";

function Statue() {
  const gltf = useGLTF("stone_statue_thinking.glb");

  useEffect(() => {
    // Traverse the scene to update materials and shadow properties
    gltf.scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        // Enable shadows for each mesh
        child.castShadow = true;
        child.receiveShadow = true;

        // Replace the material with MeshStandardMaterial
        child.material = new THREE.MeshStandardMaterial({
          color: child.material.color,
          map: child.material.map,
          roughness: 0.8,
          metalness: 0.1,
          // Copy other relevant properties if needed
        });
      }
    });
  }, [gltf]);

  return (
    <primitive
      object={gltf.scene}
      position={[0, 0, 0]}
      scale={2}
      rotation={[0, Math.PI, 0]}
    />
  );
}

export default function LoginPage() {
  const router = useRouter();
  const [camera, setCamera] = React.useState<THREE.Camera>();
  const controlsRef = useRef<any>(null);

  const changePage = () => {
    router.push("/?mode=home");
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "p") {
        if (camera && controlsRef.current) {
          console.log("Camera Position:", camera.position);
          console.log("Controls Target:", controlsRef.current.target);
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [camera]);

  return (
    <div className="relative w-screen h-screen">
      <Canvas
        camera={{ position: [0.8791, 1.317, -2.0452] }}
        className="w-full h-full touch-none"
        onCreated={({ camera }) => setCamera(camera)}
        shadows
      >
        <Environment
          files="sky_2k.hdr" // replace with your HDR path
          background={false} // This makes the HDR invisible while keeping reflections
        />
        <fog attach="fog" args={["#061025", 2, 8]} />
        <OrbitControls
          ref={controlsRef}
          target={[-0.2594, 0.7656, -0.4021]}
          enablePan={false}
          enableZoom={false}
          minPolarAngle={Math.PI / 4} // Limit vertical rotation to 45 degrees up
          maxPolarAngle={Math.PI / 2} // Limit vertical rotation to horizon
        />
        <color attach="background" args={["#061025"]} />
        <directionalLight
          position={[-10, 10, -15]}
          intensity={2}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <pointLight
          position={[0.5, 0.3, -0.8]}
          color="#A436AE"
          intensity={8}
          castShadow
          shadow-mapSize={[1024, 1024]}
        />
        <Statue />
        <WaterPlane />
      </Canvas>

      <div className="absolute fade-in top-1/2 right-1/4 transform -translate-y-1/2 translate-x-1/2">
        <div className="p-8 bg-card/80 backdrop-blur-sm rounded-lg shadow-lg w-96">
          <h1 className="text-2xl font-bold text-primary mb-6 text-center">
            Welcome to Elysia
          </h1>
          <button
            onClick={changePage}
            className="w-full btn items-center justify-center bg-background"
          >
            <p>Start Demo</p>
          </button>
        </div>
      </div>
    </div>
  );
}
