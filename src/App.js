import logo from './logo.svg';
import './App.css';
import VidStream from './templates/cam';

import { useRef, useEffect } from 'react';
import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';





export default function App() {
  useEffect(() => {
    // Set up the Intersection Observer
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          // Trigger your animation class or logic here
          entry.target.classList.add('show'); // Assuming you have a 'visible' class for animation
          observer.unobserve(entry.target); // Stop observing after it has been animated
        }
      });
    });

    // Query all hidden elements after the component mounts
    const hiddenElements = document.querySelectorAll('.hidden');
    hiddenElements.forEach((element) => observer.observe(element));

    // Cleanup the observer on component unmount
    return () => {
      hiddenElements.forEach((element) => observer.unobserve(element));
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  return (
    <div>
      <section className='hidden'>
        <h1 className='josefin-sans-400'>Whisper Vue</h1>
      </section>
      <section className='hidden'>
        <Model3D />
      </section>
      <section className='hidden'>
        <VidStream stream="http://localhost:5000/video_feed" altLabel="Live Stream"/>
      </section>
    </div>
  );
}


const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    console.log(entry);
    if(entry.isIntersecting) {
      entry.target.classList.add('show');
    } else {
      entry.target.classList.remove('show');
    }
  });
});


function Model3D() {
  const containerRef = useRef();
  const modelRef = useRef(null); // Reference to hold the loaded model
  const scrollRef = useRef(0); // To keep track of the last scroll position

  useEffect(() => {
    const container = containerRef.current;

    // Set up camera, scene, and renderer
    const camera = new THREE.PerspectiveCamera(30, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 25;

    const scene = new THREE.Scene();
    scene.background = null; // Set scene background to transparent

    // Renderer setup with transparency enabled
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Load the 3D model
    const loader = new GLTFLoader();
    loader.load(
      '/dslr_camera.glb',
      (gltf) => {
        modelRef.current = gltf.scene; // Store the loaded model
        modelRef.current.position.set(0, -1, 0); // Adjust Y position here
        scene.add(modelRef.current); // Add the model to the scene
      },
      undefined,
      (error) => {
        console.error("Error occurred while loading the model:", error);
      }
    );

    // Add ambient light to the scene
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    scene.add(ambientLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      if (modelRef.current) {
        // Apply scroll-based animation
        const scrollY = window.scrollY; // Get the current scroll position
        // Adjust model rotation based on scroll position
        modelRef.current.rotation.y = scrollY * 0.005; // Change this value for more or less sensitivity
        modelRef.current.position.y = -1 + (scrollY * 0.01); // Adjust Y position based on scroll
      }
      renderer.render(scene, camera);
    };
    animate(); // Start the animation loop

    // Handle window resize
    const handleResize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup on component unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      container.removeChild(renderer.domElement); // Remove the renderer
      renderer.dispose();
    };
  }, []); // Empty dependency array ensures this runs only once

  return <div ref={containerRef} className="container3d" style={{ width: '100%', height: '100vh' }}></div>;
}

/*
function Model3D() {
  const containerRef = useRef();
  const modelRef = useRef(null); // Reference to hold the loaded model

  useEffect(() => {
    const container = containerRef.current;

    // Set up camera, scene, and renderer
    const camera = new THREE.PerspectiveCamera(30, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 25;

    const scene = new THREE.Scene();
    scene.background = null; // Set scene background to transparent

    // Renderer setup with transparency enabled
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Load the 3D model
    const loader = new GLTFLoader();
    loader.load(
      '/dslr_camera.glb',
      (gltf) => {
        if (modelRef.current) {
          scene.remove(modelRef.current); // Remove the old model if it exists
        }
        modelRef.current = gltf.scene; // Store the loaded model
        modelRef.current.position.set(0, -1, 0); // Adjust Y position here
        modelRef.current.rotation.y = 1;
        scene.add(modelRef.current); // Add the model to the scene
      },
      undefined,
      (error) => {
        console.error("Error occurred while loading the model:", error);
      }
    );

    // Add ambient light to the scene
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    scene.add(ambientLight);

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate(); // Start the animation loop

    // Handle window resize
    const handleResize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);

    // Cleanup on component unmount
    return () => {
      window.removeEventListener('resize', handleResize);
      container.removeChild(renderer.domElement); // Remove the renderer
      renderer.dispose();
    };
  }, []); // Empty dependency array ensures this runs only once

  return <div ref={containerRef} style={{ width: '100%', height: '100vh' }}></div>;
}
  */