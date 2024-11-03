import './App.css';
import VidStream from './templates/cam';
import { useRef, useEffect } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export default function App() {
  const scrollRef = useRef();

  useEffect(() => {
    // Set up the Intersection Observer
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('show');
          observer.unobserve(entry.target);
        }
      });
    });

    const hiddenElements = document.querySelectorAll('.hidden');
    hiddenElements.forEach((element) => observer.observe(element));

    return () => {
      hiddenElements.forEach((element) => observer.unobserve(element));
    };
  }, []);

  return (
    <div>
      <nav className="navbar">
        <ul>
          <li><a href="#home">Home</a></li>
          <li><a href="#product">Product</a></li>
          <li><a href="#about">About</a></li>
        </ul>
      </nav>

      <section id="home">
        <h1></h1>
      </section>
      <section className='hidden'>
        <h1 className='josefin-sans-400'>VisionNotes</h1>
      </section>

      <section id="product">
        <h1></h1>
      </section>
      <section className='hidden'>
        <VidStream stream="http://localhost:5000/video_feed" altLabel="Live Stream" />
        {/* <div className="text-container">
          <textarea className="input-field" placeholder="Type your text here..." />
        </div> */}
      </section>

      <section id="about">
        <h1></h1>
      </section>
      <section className='hidden'>
        <h1 className="josefin-sans-400">About</h1>
        <p className="description-of-app">
        Let me introduce you to VisionNotes—this is not just another app; it’s a true companion for those who can’t see the world like most of us do. With a clever little camera that acts like an extra pair of eyes, VisionNotes scans everything around you, spotting objects and then sharing what it finds with you through clear, descriptive audio.

Picture this: you step outside, and the app tells you about the friendly dog trotting by, the busy street nearby, or even that tree with its branches swaying gently in the breeze. It’s like having a buddy beside you, narrating the scene so you never feel out of the loop. The experience is immersive, letting you connect with your surroundings in a way that feels natural and engaging.

The magic doesn’t stop there. VisionNotes is all about making everyday life a little easier. You’ll receive information about obstacles in your path, helping you to move confidently and safely. Its easy-to-use layout ensures that anyone can pick it up and get going without a hitch.

This app is all about understanding your environment in a new light—think of it as a friendly voice guiding you, ensuring that you’re not just wandering but truly experiencing the world around you. So, whether you’re at home or exploring the great outdoors, VisionNotes is here to make those moments richer and more meaningful. How could anyone resist such a delightful way to discover life’s little wonders?
        </p>
      </section>

      {/* <Model3D /> */}
    </div>
  );
}

function Model3D() {
  const containerRef = useRef();
  const modelRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    const camera = new THREE.PerspectiveCamera(30, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.z = 25;

    const scene = new THREE.Scene();
    scene.background = null;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const loader = new GLTFLoader();
    loader.load(
      '/dslr_camera.glb',
      (gltf) => {
        modelRef.current = gltf.scene;
        modelRef.current.position.set(0, -1, 0);
        scene.add(modelRef.current);
      },
      undefined,
      (error) => {
        console.error("Error occurred while loading the model:", error);
      }
    );

    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    scene.add(ambientLight);

    const animate = () => {
      requestAnimationFrame(animate);
      if (modelRef.current) {
        const scrollY = window.scrollY;
        modelRef.current.rotation.y = scrollY * 0.005;
        modelRef.current.position.y = -1 + (scrollY * 0.01);
      }
      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      container.removeChild(renderer.domElement);
      renderer.dispose();
    };
  }, []);

  return <div ref={containerRef} className="container3d" style={{ width: '100%', height: '100vh' }}></div>;
}
