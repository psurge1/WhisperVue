import logo from './logo.svg';
import './App.css';
import VidStream from './templates/cam';

import { useState } from 'react';
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';



export default function App() {
  return (
    <main>
      <h1>Whisper Vue</h1>
      <VidStream stream="http://localhost:5000/video_feed" altLabel="Live Stream"/>
      <Model3D/>
    </main>
  );
}


function Model3D() {
  const camera = THREE.PerspectiveCamera(
    10,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  )
  camera.position.z = 15;
  const scene = THREE.Scene();
  let dslr;
  const loader = new GLTFLoader();
  loader.load('/dslr_camera.glb',
    function(lmr) {
      dslr = lmr.scene;
      scene.add(dslr);
    },
    function(rfc) {},
    function (error) {}
  );
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.getElementById('container3d').appendChild(renderer.domElement);

  return (
    <div id="container3d"></div>
  )
}